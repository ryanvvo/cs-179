from model_util import *
from rbm import RBM

import torch
import torch.nn as nn
import torch.nn.functional as F

import pyro
from pyro.optim import Adam

epoch_ref = [0]
REC_PER_EPOCH = 1

class DBN(nn.Module):
    """DBN model (784) -> (512) -> (256) -> (128)"""
    def __init__(self, layer_dims=(784, 512, 256, 128), num_classes=10):
        super().__init__()

        self.layer_dims = layer_dims

        self.rbms = nn.ModuleList([
            RBM(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])

        self.classifier = nn.Linear(layer_dims[-1], num_classes)

    def forward(self, v):
        h = v
        for rbm in self.rbms:
            h = rbm.encode(h)
        return self.classifier(h)

def pretrain_dbn(dbn: DBN, epochs_per_layer: int = 5, lr: float = 1e-3):
    train_loader, _ = get_loaders()


    for layer_idx, rbm in enumerate(dbn.rbms):
        print(f"\nDBN Pretraining RBM layer {layer_idx + 1} "
              f"({rbm.v_dim} -> {rbm.h_dim})")
        optimizer = torch.optim.Adam(rbm.parameters(), lr=lr)
        for epoch in range(1, epochs_per_layer + 1):
            total_loss = 0.0
            for v, _ in train_loader:
                v = v.float()
                # propagate
                with torch.no_grad():
                    h = v
                    for prev in dbn.rbms[:layer_idx]:
                        h = prev.encode(h)

                loss = rbm.cd_loss(h, k=1)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dbn.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()

            avg = total_loss / len(train_loader.dataset)
            print(f"\tepoch {epoch} | CD loss {avg:.4f}")

def tune_dbn(meta: dict, dbn:DBN, epochs:int = 10, lr:float = 1e-3) -> DBN:
    print("\nDBN Fine-tuning with SVI (joint model + classifier)")
    train_loader, test_loader = get_loaders()
    training_acc = meta.get("training_acc", [])
    testing_acc = meta.get("testing_acc", [])
    loss_ref = meta.get("loss", [])
    print(training_acc);print(testing_acc);print(loss_ref)

    optimizer = torch.optim.Adam(dbn.parameters(), lr=lr)
    for epoch in range(epoch_ref[0] + 1, epoch_ref[0] + epochs + 1):
        print("DBN epoch", epoch)
        epoch_ref[0] = epoch
        dbn.train()
        total_loss = 0.0
        for v, y in train_loader:
            v, y = v.float(), y
            logits = dbn(v)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader.dataset)

        # eval
        dbn.eval()
        if (epoch + 1) % REC_PER_EPOCH == 0:
            print("evaluating...")
            tr_correct = tr_total = te_correct = te_total = 0

            with torch.no_grad():
                for v, y in train_loader:
                    pred = dbn(v).argmax(dim=-1)
                    tr_correct += (pred == y).sum().item()
                    tr_total += y.size(0)
                for v, y in test_loader:
                    pred = dbn(v).argmax(dim=-1)
                    te_correct += (pred == y).sum().item()
                    te_total += y.size(0)

            tr_acc = 100.0 * tr_correct / tr_total
            te_acc = 100.0 * te_correct / te_total

            testing_acc.append((epoch+1, te_acc))
            training_acc.append((epoch+1, tr_acc))
            loss_ref.append((epoch+1, avg))
            meta["training_acc"]=training_acc; meta["testing_acc"]=testing_acc; meta['loss']=loss_ref
            print(f"CD {avg:.4f} | train acc {tr_acc:.2f} % "
                  f"| test acc {te_acc:.2f} %")

        dbn.train()


    return dbn

if __name__ == '__main__':
    #hyperparams
    epochs_per_layer = 20
    lr = 1e-3
    epochs = 200

    pyro.clear_param_store()
    dbn = DBN()

    payload = load_model("deep-belief-network", dbn)
    meta = payload.get("meta", {})
    install_interrupt_save("deep-belief-network", dbn, epoch_ref, meta)
    if not payload:
        pretrain_dbn(dbn, epochs_per_layer=epochs_per_layer, lr=lr)
    else:
        print("Pretraining skipped. Tuning...")
    epoch_ref[0] = payload.get("epoch", 0)
    tune_dbn(meta, dbn, epochs=epochs, lr=lr)

    save_model("deep-belief-network", dbn, epoch_ref[0], meta)
