from model_util import *

import torch
import torch.nn as nn
import torch.optim as optim

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
    install_interrupt_save("linear-classifier", model, epoch_ref)
    payload = load_model("linear-classifier", model=model)

    # training
    epoch_ref[0] = payload.get("epoch", 0)
    starting_epoch = epoch_ref[0]
    epochs = 10

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

        print(
            f"Epoch {epoch+1}/{starting_epoch + epochs} | "
            f"Loss: {running_loss/len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}%"
        )

    # evaluation
    model.eval()

    correct = 0
    total = 0

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
    save_model("linear-classifier", model=model, epoch = starting_epoch + epochs)