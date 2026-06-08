from model_util import *
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

epoch_ref = [0]
REC_PER_EPOCH = 1

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

    def sample_h(self, v):
        probs = self.p_h_given_v(v)
        return torch.bernoulli(probs), probs

    def sample_v(self, h):
        probs = self.p_v_given_h(h)
        return torch.bernoulli(probs), probs

    def gibbs_k(self, v0, k=1):
        """ Run k Gibbs steps beginning at v0. Returns: vk phk """
        vk = v0
        for _ in range(k):
            hk, _ = self.sample_h(vk)
            vk, _ = self.sample_v(hk)
        phk = self.p_h_given_v(vk)
        return vk.detach(), phk.detach()

    def cd_loss(self, v, k=1):
        vk, _ = self.gibbs_k(v, k)

        return self.free_energy(v).mean() - self.free_energy(vk).mean()
    @torch.no_grad()
    def reconstruct(self, v):
        """ Deterministic reconstruction. """
        h = self.p_h_given_v(v)
        return self.p_v_given_h(h)

    @torch.no_grad()
    def reconstruction_error(self, v):
        recon = self.reconstruct(v)
        return nn.functional.binary_cross_entropy(recon, v, reduction="mean")

    @torch.no_grad()
    def encode(self, v: torch.Tensor) -> torch.Tensor:
        """Return mean-field hidden activations."""
        return self.p_h_given_v(v)

    def free_energy(self, v):
        """Free energy of visible state for monitoring training. """
        vbias_term = torch.matmul(v, self.b_v)
        hidden_term = torch.log1p(torch.exp(v @ self.W + self.b_h)).sum(dim=1)
        return -(vbias_term + hidden_term)

def train_rbm(meta:dict, rbm: RBM, epochs:int = 10, h_dim:int =256,lr:float =1e-3) -> RBM:
    train_loader, test_loader = get_loaders()
    training_acc = meta.get("training_acc", [])
    testing_acc = meta.get("testing_acc", [])
    loss_ref = meta.get("loss", [])
    print(training_acc); print(testing_acc); print(loss_ref)

    # training
    optimizer = torch.optim.Adam(rbm.parameters(), lr=1e-3)

    for epoch in range(epoch_ref[0] + 1, epoch_ref[0] + epochs + 1):
        print("RBM epoch", epoch)
        epoch_ref[0] = epoch
        total_loss = 0.0
        for v, _ in train_loader:
            v = v.float()
            optimizer.zero_grad()
            loss = rbm.cd_loss(v, k=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rbm.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        avg = total_loss / len(train_loader.dataset)

        if (epoch + 1) % REC_PER_EPOCH == 0:
            print("evaluating...")
            tr_correct = tr_total = te_correct = te_total = 0
            for v, _ in train_loader:
                v = v.float()
                recon = rbm.reconstruct(v)
                pred = (recon > 0.5).float()
                tr_correct += (pred == v).sum().item()
                tr_total += v.numel()
            for v, _ in test_loader:
                v = v.float()
                recon = rbm.reconstruct(v)
                pred = (recon > 0.5).float()
                te_correct += (pred == v).sum().item()
                te_total += v.numel()

            tr_acc = 100.0 * tr_correct / tr_total
            te_acc = 100.0 * te_correct / te_total

            testing_acc.append((epoch + 1, te_acc))
            training_acc.append((epoch + 1, tr_acc))
            loss_ref.append((epoch + 1, avg))
            meta["training_acc"] = training_acc;meta["testing_acc"] = testing_acc;meta['loss'] = loss_ref
            print(f"CD Loss {avg:.4f} | train acc {tr_acc:.2f} % "
                  f"| test acc {te_acc:.2f} %")

    return rbm

if __name__ == '__main__':
    #hyper params
    epochs = 100
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
