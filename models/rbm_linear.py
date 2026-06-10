from model_util import *
from rbm import RBM

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
REC_PER_EPOCH = 1
epoch_ref = [0]


class RBMLinearClassifier(nn.Module):
	def __init__(self, h_dim: int = 256):
		super().__init__()
		self.classifier = nn.Linear(h_dim, NUM_CLASSES)

	def forward(self, h: torch.Tensor):
		return self.classifier(h)

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	rbm = RBM()
	clf = RBMLinearClassifier(h_dim=rbm.h_dim)

	rbm.to(device)
	clf.to(device)

	# criterion / optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(clf.parameters(), lr=0.01)

	# data
	train_loader, test_loader = get_loaders()

	payload_rbm = load_model("restricted-boltzmann-model", model=rbm)
	payload_clf = load_model("rbm-linear-classifier", model=clf)

	# metadata bookkeeping
	meta = payload_clf.get("meta", {})
	training_acc = meta.get("training_acc", [])
	testing_acc = meta.get("testing_acc", [])
	loss_ref = meta.get("loss", [])
	print(training_acc); print(testing_acc); print(loss_ref)

	install_interrupt_save("rbm-linear-classifier", clf, epoch_ref, meta)

	epoch_ref[0] = payload_clf.get("epoch", 0)
	starting_epoch = epoch_ref[0]
	epochs = 195

	rbm.eval()
	for epoch in range(starting_epoch, starting_epoch + epochs):
		epoch_ref[0] = epoch
		clf.train()

		running_loss = 0.0
		correct = 0
		total = 0

		for images, labels in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			# obtain RBM encoding
			with torch.no_grad():
				h = rbm.encode(images)  # shape (batch, h_dim)
			h = h.to(device)

			optimizer.zero_grad()
			outputs = clf(h)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

		train_acc = 100.0 * correct / total if total > 0 else 0.0
		if (epoch + 1) % REC_PER_EPOCH == 0:
			print("Logging training_acc and loss...")
			training_acc.append((epoch + 1, train_acc))
			loss_ref.append((epoch + 1, running_loss / len(train_loader)))

			# evaluate on test set
			clf.eval()
			te_correct = te_total = 0
			with torch.no_grad():
				for images, labels in test_loader:
					images = images.to(device)
					labels = labels.to(device)
					h = rbm.encode(images).to(device)
					outputs = clf(h)
					_, predicted = outputs.max(1)
					te_total += labels.size(0)
					te_correct += predicted.eq(labels).sum().item()

			test_acc = 100.0 * te_correct / te_total if te_total > 0 else 0.0
			print(f"Test Accuracy: {test_acc:.2f}%")
			testing_acc.append((epoch + 1, test_acc))
			meta["training_acc"] = training_acc; meta["testing_acc"] = testing_acc; meta['loss'] = loss_ref

		print(
			f"Epoch {epoch+1}/{starting_epoch + epochs} | "
			f"Loss: {running_loss/len(train_loader):.4f} | "
			f"Train Acc: {train_acc:.2f}%"
		)

	# finalize
	meta["training_acc"] = training_acc; meta["testing_acc"] = testing_acc; meta['loss'] = loss_ref
	save_model("rbm-linear-classifier", model=clf, epoch=starting_epoch + epochs, meta=meta)

