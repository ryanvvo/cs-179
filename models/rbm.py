from model_util import *
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

epoch_ref = [0]

# note: the model is not a predictor. it instead produces a representation of the data
# through its hidden layers in order to aid other classifiers.
class RBM(nn.Module):
    """
    Binary RBM:
        - visible (v-dim) | 784 pixels
        - hidden (h-dim) | 256 default
    """
    def __init__(self, v_dim:int = IMG_DIM, h_dim:int = 256):
        super().__init__()
        self.v_dim = v_dim
        self.h_dim = h_dim

        # learnable parameters
        self.W = nn.Parameter(torch.randn(v_dim, h_dim) * 0.01)
        self.b_v = nn.Parameter(torch.zeros(v_dim)) # biases for visible
        self.b_h = nn.Parameter(torch.zeros(h_dim)) # biases for hidden

    def p_h_given_v(self, v:torch.Tensor) -> torch.Tensor:
        """Probability of hidden given the visible. shape: (..., H)"""
        return torch.sigmoid(v @ self.W + self.b_h)

    def p_v_given_h(self, h: torch.Tensor) -> torch.Tensor:
        """Probability of visible given the hidden. shape: (..., V)"""
        return torch.sigmoid(h @ self.W.T + self.b_v)

    def gibbs_sample(self, v:torch.Tensor, k:int = 1):
        """
        Run k steps of block Gibbs sampling starting from v.
        Returns (v_k, h_k, p_h_k) where p_h_k are the final
        hidden probabilities (used for contrastive divergence).
        """
        v_k = v.clone()
        for _ in range(k):
            p_h = self.p_h_given_v(v_k)
            h_k = torch.bernoulli(p_h)
            p_v = self.p_v_given_h(h_k)
            v_k = torch.bernoulli(p_v)
        p_h_k = self.p_h_given_v(v_k)
        return v_k, p_h_k

    def model(self, v:torch.Tensor):
        """
        v ~ Bernoulli(sigmoid(W h + b_v))
        We register W, b_v, b_h in the Pyro param store so SVI
        can optimise them alongside the variational parameters.
        """
        pyro.module("rbm", self)

        with pyro.plate("data", v.shape[0]):
            # Sample hidden units from prior
            h_prior = torch.full((v.shape[0], self.h_dim), 0.5, device=v.device)
            h = pyro.sample("h", dist.Bernoulli(h_prior).to_event(1))

            # Likelihood: reconstruct visible
            v_probs = self.p_v_given_h(h)
            pyro.sample("v", dist.Bernoulli(v_probs).to_event(1), obs=v)

    def guide(self, v:torch.Tensor):
        """
        Mean-field variational posterior: q(h | v) = Bernoulli(sigmoid(W^T v + b_h))
        """
        pyro.module("rbm", self)

        with pyro.plate("data", v.shape[0]):
            q_h = self.p_h_given_v(v)
            pyro.sample("h", dist.Bernoulli(q_h).to_event(1))

    def cd_loss(self, v: torch.Tensor, k: int = 1) -> torch.Tensor:
        """
        Contrastive divergence loss (CD-k).
        """
        p_h0 = self.p_h_given_v(v) # positive phase
        v_k, p_hk = self.gibbs_sample(v, k) # negative phase

        # Gradients are proportional to <v h^T>_data - <v h^T>_model
        pos = torch.einsum("bi,bj->ij", v, p_h0) / v.shape[0]
        neg = torch.einsum("bi,bj->ij", v_k, p_hk) / v.shape[0]
        return -(pos - neg).sum() # scalar loss

    def sample_h(self, v: torch.Tensor) -> torch.Tensor:
        """Stochastic hidden sample from q(h|v)."""
        p_h = self.p_h_given_v(v)
        return torch.bernoulli(p_h)

    def sample_v(self, h: torch.Tensor) -> torch.Tensor:
        """Stochastic visible sample from p(v|h)."""
        p_v = self.p_v_given_h(h)
        return torch.bernoulli(p_v)

    @torch.no_grad()
    def encode(self, v: torch.Tensor) -> torch.Tensor:
        """Return mean-field hidden activations."""
        return self.p_h_given_v(v)

def train_rbm(meta:dict, rbm: RBM, epochs:int = 10, h_dim:int =256,lr:float =1e-3) -> RBM:
    train_loader, test_loader = get_loaders()
    training_acc = meta.get("training_acc", [])
    testing_acc = meta.get("testing_acc", [])
    loss_ref = meta.get("loss", [])
    print(training_acc); print(testing_acc); print(loss_ref)

    svi = SVI(rbm.model, rbm.guide, Adam({"lr": lr}), loss=Trace_ELBO())

    for epoch in range(epoch_ref[0] + 1, epoch_ref[0] + epochs + 1):
        epoch_ref[0] = epoch
        total_loss = 0.0
        for v, _ in train_loader:
            total_loss += svi.step(v)
        avg = total_loss / len(train_loader.dataset)
        print("RBM epoch", epoch)
        if (epoch + 1) % 10 == 0:
            print("evaluating...")
            tr_correct = tr_total = te_correct = te_total = 0
            for v, _ in train_loader:
                v = v.float()
                h = rbm.sample_h(v)
                v_recon = rbm.sample_v(h)

                pred = (v_recon > 0.5).float()
                tr_correct += (pred == v).sum().item()
                tr_total += v.numel()
            for v, _ in test_loader:
                v = v.float()
                h = rbm.sample_h(v)
                v_recon = rbm.sample_v(h)

                pred = (v_recon > 0.5).float()
                te_correct += (pred == v).sum().item()
                te_total += v.numel()

            tr_acc = 100.0 * tr_correct / tr_total
            te_acc = 100.0 * te_correct / te_total

            testing_acc.append((epoch + 1, te_acc))
            training_acc.append((epoch + 1, tr_acc))
            loss_ref.append((epoch + 1, avg))
            meta["training_acc"] = training_acc;meta["testing_acc"] = testing_acc;meta['loss'] = loss_ref
            print(f"ELBO {avg:.4f} | train acc {tr_acc:.2f} % "
                  f"| test acc {te_acc:.2f} %")

    return rbm

if __name__ == '__main__':
    #hyper params
    epochs = 10
    hidden_dim = 256
    lr = 1e-3

    pyro.clear_param_store()
    rbm_model = RBM(IMG_DIM, hidden_dim)
    payload = load_model("restricted-boltzmann-model", rbm_model)
    meta = payload.get("meta", {})

    install_interrupt_save("restricted-boltzmann-model", rbm_model, epoch_ref, meta)

    epoch_ref[0] = payload.get("epoch", 0)

    train_rbm(meta, rbm_model, epochs, hidden_dim, lr)

    save_model("restricted-boltzmann-model", rbm_model, epoch_ref[0], meta)
