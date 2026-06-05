from model_util import *
from rbm import RBM

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam

epoch_ref = [0]
class DBN(nn.Module):
    """
    3-layer DBN:
      v (784) -> h1 (512) -> h2 (256) -> h3 (128) -> softmax (10)

    Greedy pretraining trains three RBMs in sequence.
    Fine-tuning uses the deterministic (mean-field) forward pass.
    """
    LAYER_DIMS = [IMG_DIM, 512, 256, 128]

    def __init__(self):
        super().__init__()
        self.rbms = nn.ModuleList([
            RBM(v_dim=self.LAYER_DIMS[i], h_dim=self.LAYER_DIMS[i + 1])
            for i in range(len(self.LAYER_DIMS) - 1)
        ])
        self.classifier = nn.Linear(self.LAYER_DIMS[-1], NUM_CLASSES)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Propagate through RBM stack using mean-field activations,
        then apply the softmax head. Returns log-probabilities.
        """
        h = v
        for rbm in self.rbms:
            h = rbm.p_h_given_v(h)
        return F.log_softmax(self.classifier(h), dim=-1)

    def model(self, v: torch.Tensor, y: torch.Tensor):
        """
        Hierarchical generative model:
            h3 ~ Bernoulli(0.5)^128
            h2 ~ Bernoulli(sigmoid(W3 h3 + b_v3))^256
            h1 ~ Bernoulli(sigmoid(W2 h2 + b_v2))^512
            v  ~ Bernoulli(sigmoid(W1 h1 + b_v1))^784
            y  ~ Categorical(softmax(W_cls h3))
        """
        pyro.module("dbn", self)
        batch = v.shape[0]
        device = v.device

        with pyro.plate("data", batch):
            # Top-level hidden layer (prior)
            h3 = pyro.sample(
                "h3",
                dist.Bernoulli(
                    torch.full((batch, self.LAYER_DIMS[3]), 0.5, device=device)
                ).to_event(1),
            )
            # h2 | h3
            h2 = pyro.sample(
                "h2",
                dist.Bernoulli(self.rbms[2].p_v_given_h(h3)).to_event(1),
            )
            # h1 | h2
            h1 = pyro.sample(
                "h1",
                dist.Bernoulli(self.rbms[1].p_v_given_h(h2)).to_event(1),
            )
            # v | h1
            pyro.sample(
                "v",
                dist.Bernoulli(self.rbms[0].p_v_given_h(h1)).to_event(1),
                obs=v,
            )
            # label | h3
            logits = self.classifier(h3.float())
            pyro.sample("y", dist.Categorical(logits=logits), obs=y)

    def guide(self, v: torch.Tensor, y: torch.Tensor):
        """
        Bottom-up mean-field inference (amortised):
            q(h1|v) = Bernoulli(p_h_given_v_rbm1(v))
            q(h2|h1) = Bernoulli(p_h_given_v_rbm2(h1))
            q(h3|h2) = Bernoulli(p_h_given_v_rbm3(h2))
        """
        pyro.module("dbn", self)
        batch = v.shape[0]

        with pyro.plate("data", batch):
            h1_probs = self.rbms[0].p_h_given_v(v)
            h1 = pyro.sample("h1", dist.Bernoulli(h1_probs).to_event(1))

            h2_probs = self.rbms[1].p_h_given_v(h1)
            h2 = pyro.sample("h2", dist.Bernoulli(h2_probs).to_event(1))

            h3_probs = self.rbms[2].p_h_given_v(h2)
            pyro.sample("h3", dist.Bernoulli(h3_probs).to_event(1))

def pretrain_dbn(dbn: DBN, epochs_per_layer: int = 5, lr: float = 1e-3):
    train_loader, _ = get_loaders()

    for layer_idx, rbm in enumerate(dbn.rbms):
        print(f"\nDBN Pretraining RBM layer {layer_idx + 1} "
              f"({rbm.v_dim} → {rbm.h_dim})")

        svi = SVI(rbm.model, rbm.guide,
                  Adam({"lr": lr}),
                  loss=Trace_ELBO())

        for epoch in range(1, epochs_per_layer + 1):
            total_loss = 0.0
            for v, _ in train_loader:
                # Pass v through already-trained lower layers
                with torch.no_grad():
                    h = v
                    for prev_rbm in dbn.rbms[:layer_idx]:
                        h = torch.bernoulli(prev_rbm.p_h_given_v(h))
                total_loss += svi.step(h)
            avg = total_loss / len(train_loader.dataset)
            print(f"\tepoch {epoch} | ELBO loss {avg:.4f}")

def tune_dbn(dbn:DBN, epochs:int = 10, lr:float = 1e-3) -> DBN:
    print("\nDBN Fine-tuning with SVI (joint model + classifier)")
    train_loader, test_loader = get_loaders()

    svi = SVI(dbn.model, dbn.guide,
              ClippedAdam({"lr": lr, "clip_norm": 10.0}),
              loss=Trace_ELBO())

    for epoch in range(epoch_ref[0] + 1, epoch_ref[0] + epochs + 1):
        epoch_ref[0] = epoch
        # SVI Step
        total_loss = 0.0
        for v, y in train_loader:
            total_loss += svi.step(v, y)
        avg = total_loss / len(train_loader.dataset)

        # Accuracy
        correct = total = 0
        dbn.eval()
        with torch.no_grad():
            for v, y in test_loader:
                preds = dbn(v).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        dbn.train()

        print(f"DBN epoch {epoch} | ELBO {avg:.4f} "
              f"| test acc {correct / total:.3%}")

    return dbn

if __name__ == '__main__':
    #hyperparams
    epochs_per_layer = 5
    lr = 1e-3
    epochs = 10

    pyro.clear_param_store()
    dbn = DBN()

    install_interrupt_save("deep-belief-network", dbn, epoch_ref)
    payload = load_model("saved_models/deep-belief-network.pt", dbn)
    if not payload:
        pretrain_dbn(dbn, epochs_per_layer=epochs_per_layer, lr=lr)
    else:
        print("Pretraining skipped. Tuning...")
    epoch_ref[0] = payload.get("epoch", 0)
    tune_dbn(dbn, epochs=epochs, lr=lr)

    save_model("deep-belief-network", dbn, epoch_ref[0])
