import pyro, torch, signal
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(MODULE_DIR, "saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_CLASSES = 10
IMG_DIM = 784 # 28 × 28
pyro.set_rng_seed(123)


BATCH_SIZE = 128

tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)),          # flatten to (784,)
    transforms.Lambda(lambda x: (x > 0.5).float()),   # binary for RBM / DBN
])

def get_loaders():
    """ Loads data if not already loaded """
    train = datasets.FashionMNIST("data", train=True,  download=True, transform=tfm)
    test  = datasets.FashionMNIST("data", train=False, download=True, transform=tfm)
    return (DataLoader(train, batch_size=BATCH_SIZE, shuffle=True),
            DataLoader(test,  batch_size=BATCH_SIZE))

def save_model(name:str, model:"nn.Module | None", epoch:int, meta: dict | None = None) -> str:
    """
    Save model weights + Pyro param store to
    SAVE_DIR/<name>_epoch<N>.pt  Returns the path written.
    model is for nn.Module; you only need to save params for NB model.
    """
    path = os.path.join(SAVE_DIR, f"{name}.pt")
    payload: dict = {
        "epoch": epoch,
        "param_store": pyro.get_param_store().get_state(),
        "meta": meta or {},
    }
    if model is not None:
        payload["model_state"] = model.state_dict()
    torch.save(payload, path)
    print(f"model saved  -> {path}")
    return path


def load_model(name: str, model: "nn.Module | None" = None,) -> dict:
    """
    Restore weights + Pyro param store from a checkpoint file.
    Returns the full payload dict so callers can read epoch / meta.
    """
    path = os.path.join(SAVE_DIR, f"{name}.pt")
    print("Loading from", path)
    try:
        payload = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print("No saved model found. Continuing...")
        return {}
    pyro.get_param_store().set_state(payload["param_store"])
    if model is not None and "model_state" in payload:
        model.load_state_dict(payload["model_state"])
    print(f" model loaded <- {path}  (epoch {payload['epoch']})")
    return payload


def install_interrupt_save(name: str, model: "nn.Module | None", epoch_ref: list, meta: dict):
    """
    Register a SIGINT (Ctrl+C) handler that saves a checkpoint
    before re-raising KeyboardInterrupt.
    epoch_ref is a one-element list so the closure sees the live value.
    """
    original = signal.getsignal(signal.SIGINT)

    def _handler(sig, frame):
        print(f"\nsaving {name} at epoch {epoch_ref[0]} ...")
        save_model(name, model, epoch_ref[0], meta=meta)
        signal.signal(signal.SIGINT, original)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)

