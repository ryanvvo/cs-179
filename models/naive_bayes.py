from model_util import *

import torch
from torch.utils.data import DataLoader

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

epoch_ref = [0]
install_interrupt_save("naive-bayes-model", None, epoch_ref)
class NaiveBayesModel:
    """
    Bayesian Naive Bayes with:
        - Dirichlet prior on class probabilities pi
        - Beta prior on per-class, per-pixel probabilities theta_{c,i}

    Parameters live in the Pyro param store; training runs SVI.
    """

    def __init__(
            self,
            num_classes:int = NUM_CLASSES,
            img_dim:int = IMG_DIM,
            alpha_pi:float = 1.0,
            alpha_theta:float = 1.0,
            beta_theta:float = 1.0
    ):
        self.C = num_classes
        self.D = img_dim
        self.alpha_pi = alpha_pi
        self.alpha_theta = alpha_theta
        self.beta_theta = beta_theta

    def model(self, x:torch.Tensor, y:torch.Tensor | None = None):
        """
        p(pi, theta, y, x):
          pi     ~ Dirichlet(α_pi · 1_C)
          theta_{c} ~ Beta(alpha, beta)^D for each class c
          y_n    ~ Categorical(pi)
          x_ni   ~ Bernoulli(theta_{y_n, i})
        """
        # Class prior
        pi = pyro.sample(
            "pi",
            dist.Dirichlet(
                torch.full((self.C,), self.alpha_pi)
            ),
        )

        # Per-class pixel probabilities: shape (C, D)
        with pyro.plate("classes", self.C):
            theta = pyro.sample(
                "theta",
                dist.Beta(
                    torch.full((self.D,), self.alpha_theta),
                    torch.full((self.D,), self.beta_theta),
                ).to_event(1),
            ) # (C, D)

        # Observe data
        with pyro.plate("data", x.shape[0]):
            # Sample class label
            y_obs = pyro.sample("y", dist.Categorical(pi), obs=y)

            # Pixel likelihood P(x | θ_{y})
            theta_n = theta[y_obs] # (N, D)
            pyro.sample(
                "x",
                dist.Bernoulli(theta_n).to_event(1),
                obs=x,
            )

    def guide(self, x: torch.Tensor, y: torch.Tensor | None = None):
        """
        Mean-field variational posterior:
          q(pi) = Dirichlet(α_hat_pi)
          q(theta_{c})= Beta(a_hat_{c,i}, b_hat_{c,i}) for each (c, i)
        """
        # Variational Dirichlet for pi
        alpha_q = pyro.param(
            "alpha_q",
            torch.ones(self.C) * self.alpha_pi,
            constraint=dist.constraints.positive,
        )
        pyro.sample("pi", dist.Dirichlet(alpha_q))

        # Variational Beta for theta - shape (C, D) each
        a_q = pyro.param(
            "a_q",
            torch.ones(self.C, self.D) * self.alpha_theta,
            constraint=dist.constraints.positive,
        )
        b_q = pyro.param(
            "b_q",
            torch.ones(self.C, self.D) * self.beta_theta,
            constraint=dist.constraints.positive,
        )
        with pyro.plate("classes", self.C):
            pyro.sample("theta", dist.Beta(a_q, b_q).to_event(1))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels using the posterior mean of theta and pi:
            y_hat = argmax_c log pi_c + sum(x_i log theta_{c,i} + (1-x_i) log(1-theta_{c,i}))
        """
        # Posterior means
        a_q = pyro.param("a_q") # (C, D)
        b_q = pyro.param("b_q") # (C, D)
        theta_mean = a_q / (a_q + b_q) # (C, D)

        alpha_q = pyro.param("alpha_q") # (C,)
        pi_mean = alpha_q / alpha_q.sum() # (C,)

        # Log-likelihood for each class (N, C)
        x_ = x.unsqueeze(1) # (N, 1, D)
        theta_ = theta_mean.unsqueeze(0) # (1, C, D)

        log_lik = (
            x_ * torch.log(theta_ + 1e-9)
            + (1 - x_) * torch.log(1 - theta_ + 1e-9)
        ).sum(-1) # (N, C)

        log_prior = torch.log(pi_mean + 1e-9).unsqueeze(0) # (1, C)
        log_post  = log_lik + log_prior # (N, C)

        return log_post.argmax(dim=-1)

    @staticmethod
    def fit_mle(train_loader: DataLoader, laplace: float = 1.0):
        """
        Maximum-likelihood estimation with Laplace smoothing.
        Returns (log_pi, log_theta, log_one_minus_theta) ready
        for fast prediction without the Pyro param store.
        """
        counts = torch.zeros(NUM_CLASSES, IMG_DIM)
        class_n = torch.zeros(NUM_CLASSES)

        for x, y in train_loader:
            for c in range(NUM_CLASSES):
                mask = (y == c)
                counts[c]   += x[mask].sum(0)
                class_n[c]  += mask.sum()

        theta = (counts + laplace) / (class_n.unsqueeze(1) + 2 * laplace)
        pi = (class_n + laplace) / (class_n.sum() + NUM_CLASSES * laplace)
        return torch.log(pi), torch.log(theta), torch.log(1 - theta)

def train_naive_bayes(epochs:int = 5, lr:float = 5e-3) -> NaiveBayesModel:
    train_loader, test_loader = get_loaders()

    nb  = NaiveBayesModel()
    svi = SVI(nb.model, nb.guide, Adam({"lr": lr}), loss=Trace_ELBO())

    for epoch in range(epoch_ref[0] + 1, epoch_ref[0] + epochs + 1):
        epoch_ref[0] = epoch
        total_loss = 0.0
        for x, y in train_loader:
            total_loss += svi.step(x, y)
        avg = total_loss / len(train_loader.dataset)

        # Accuracy
        correct = total = 0
        for x, y in test_loader:
            preds = nb.predict(x)
            correct += (preds == y).sum().item()
            total   += y.size(0)

        print(f"NB epoch {epoch} | ELBO {avg:.4f} "
              f"| test acc {correct / total:.3%}")

    return nb

if __name__ == '__main__':
    pyro.clear_param_store()
    payload = load_model("naive-bayes-model", None)
    epoch_ref[0] = payload.get("epoch", 0)
    # Hyper parameters
    epochs = 5
    lr = 5e-3
    nb = train_naive_bayes(epoch_ref[0], epochs, lr)
    save_model("naive-bayes-model", None, epoch_ref[0])