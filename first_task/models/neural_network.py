import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .interface import MnistClassifierInterface


class FeedForwardNN(nn.Module):
    """
    Simple fully connected neural network.

    Architecture:
    784 → 128 → 64 → 10
    """

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)

class NeuralNetworkMnistClassifier(MnistClassifierInterface):

    def __init__(self, epochs = 10, batch_size = 128, patience = 3):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN().to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.history = None

    def train(self, X_train, y_train, X_val=None, y_val=None):

        X_train = X_train.astype(np.float32) / 255.0
        X_train = X_train.reshape(len(X_train), -1)

        X_train = torch.from_numpy(X_train).to(self.device)
        y_train = torch.from_numpy(y_train).long().to(self.device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_available = X_val is not None and y_val is not None

        if val_available:
            X_val = X_val.astype(np.float32) / 255.0
            X_val = X_val.reshape(len(X_val), -1)

            X_val = torch.from_numpy(X_val).to(self.device)
            y_val = torch.from_numpy(y_val).long().to(self.device)

            best_val_loss = float("inf")
            early_counter = 0

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": []
        }

        for epoch in range(self.epochs):

            self.model.train()

            total_loss = 0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:

                outputs = self.model(X_batch)
                loss = self.loss_fn(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * y_batch.size(0)

                _, pred = torch.max(outputs, 1)

                correct += (pred == y_batch).sum().item()
                total += y_batch.size(0)

            train_loss = total_loss / total
            train_acc = correct / total

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_available:
                self.model.eval()
                with torch.no_grad():

                    val_outputs = self.model(X_val)
                    val_loss = self.loss_fn(val_outputs, y_val).item()
                    _, val_pred = torch.max(val_outputs, 1)
                    val_acc = (val_pred == y_val).sum().item() / len(y_val)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                print(f"Epoch {epoch+1}: train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_counter = 0
                else:
                    early_counter += 1

                    if early_counter >= self.patience:
                        print("Early stopping triggered.")
                        break

    def predict(self, X):

        if len(X) == 0:
            return np.array([])

        X = X.astype(np.float32) / 255.0
        X = X.reshape(len(X), -1)
        X = torch.from_numpy(X).to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X)

        return torch.argmax(outputs, dim=1).cpu().numpy()

    def get_metrics(self):
        return self.history