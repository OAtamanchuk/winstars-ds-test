import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .interface import MnistClassifierInterface

class CNN(nn.Module):
    """
    Convolutional Neural Network for MNIST classification.

    CNN is well-suited for image tasks because convolutional layers
    preserve spatial structure of the image.
    Added BatchNorm and Dropout for better training stability and regularization.
    """

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # First convolution block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  # Added BatchNorm for stability
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),  # Light dropout after pool

            # Second convolution block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Stronger dropout in FC

            nn.Linear(128, 10)
        )

    def forward(self, x):
        if x.numel() == 0:  # Handle empty input (fix for edge case)
            return torch.empty((0, 10), device=x.device, dtype=torch.float32)

        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

class CNNMnistClassifier(MnistClassifierInterface):
    """
    CNN-based classifier for MNIST.

    Unlike classical ML models, CNNs operate directly on
    2D image structures without flattening.
    """

    def __init__(self, epochs = 10, batch_size = 128, patience = 3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNN().to(self.device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.history = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None) -> None:
        # Normalize and preprocess train
        X_train = X_train.astype(np.float32) / 255.0
        X_train_tensor = torch.from_numpy(X_train).unsqueeze(1).to(self.device)
        y_train_tensor = torch.from_numpy(y_train).long().to(self.device)

        # DataLoader for train
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Preprocess val if provided
        val_available = X_val is not None and y_val is not None
        if val_available:
            X_val = X_val.astype(np.float32) / 255.0
            X_val_tensor = torch.from_numpy(X_val).unsqueeze(1).to(self.device)
            y_val_tensor = torch.from_numpy(y_val).long().to(self.device)
            best_val_loss = float('inf')
            early_stop_counter = 0
        else:
            print("No validation set provided; early stopping disabled.")

        # History for metrics
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            # Train phase
            self.model.train()
            epoch_train_loss = 0.0
            correct_train = 0
            total_train = 0
            for batch_X, batch_y in dataloader:
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_train_loss += loss.item() * batch_y.size(0)
                _, predicted = torch.max(outputs, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            avg_train_loss = epoch_train_loss / total_train
            train_acc = correct_train / total_train
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)

            # Val phase
            if val_available:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = self.loss_fn(val_outputs, y_val_tensor).item()
                    _, val_pred = torch.max(val_outputs, 1)
                    val_acc = (val_pred == y_val_tensor).sum().item() / len(y_val_tensor)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}.")
                        break
            else:
                print(f"Epoch {epoch+1}/{self.epochs} | Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32) / 255.0
        X_tensor = torch.from_numpy(X).unsqueeze(1).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)

        return torch.argmax(outputs, dim=1).cpu().numpy()
    
    def get_metrics(self):
        return self.history