from model_util import *

import torch
import torch.nn as nn
import torch.optim as optim
REC_PER_EPOCH = 1
epoch_ref = [0]

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = x.view(x.size(0), -1)  # flatten to (batch_size, 784)
        return self.fc(x)          # logits

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LinearClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    train_loader, test_loader = get_loaders()
    payload = load_model("linear-classifier", model=model)

    meta = payload.get("meta", {})
    training_acc = meta.get("training_acc", [])
    testing_acc = meta.get("testing_acc", [])
    loss_ref = meta.get("loss", [])
    print(training_acc);print(testing_acc);print(loss_ref)
    install_interrupt_save("linear-classifier", model, epoch_ref, meta)

    #Training
    epoch_ref[0] = payload.get("epoch", 0)
    starting_epoch = epoch_ref[0]
    epochs = 200

    for epoch in range(starting_epoch , starting_epoch + epochs):
        epoch_ref[0] = epoch
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        if (epoch+1) % REC_PER_EPOCH == 0:
            print("Logging training_acc and loss...")
            training_acc.append((epoch+1, train_acc))
            loss_ref.append((epoch+1, running_loss/len(train_loader)))

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    _, predicted = outputs.max(1)

                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            test_acc = 100.0 * correct / total
            print(f"Test Accuracy: {test_acc:.2f}%")
            testing_acc.append((epoch + 1, test_acc))
            meta["training_acc"]=training_acc; meta["testing_acc"]=testing_acc; meta['loss']=loss_ref

        print(
            f"Epoch {epoch+1}/{starting_epoch + epochs} | "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

    # evaluation
    model.eval()

    correct = 0
    total = 0


    meta["training_acc"]=training_acc; meta["testing_acc"]=testing_acc; meta['loss']=loss_ref
    save_model("linear-classifier", model=model, epoch = starting_epoch + epochs, meta=meta)