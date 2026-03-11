import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# Default training hyperparameters
DEFAULT_EPOCHS = 15  # Number of training epochs
DEFAULT_BATCH_SIZE = 32  # Batch size for data loading
DEFAULT_LR = 0.0005  # Learning rate for optimizer
DEFAULT_PATIENCE = 4  # Early stopping patience

MODEL_PATH = "models/vision_model.pth"  # Path to save the trained model


def get_dataloaders(data_dir, batch_size):
    """
    Create training and validation DataLoaders from an image dataset.
    """
    # Define image transformations: resize, convert to tensor, normalize
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    # Load dataset from ImageFolder
    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # Split dataset into train (80%) and validation (20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )

    return train_loader, val_loader, dataset.classes


def build_model(num_classes):
    """
    Build a ResNet18 model with a custom classification head.
    """
    model = models.resnet18(weights="DEFAULT")

    # Replace the final fully connected layer with custom head
    model.fc = nn.Sequential(
        nn.Dropout(0.3),  # Dropout for regularization
        nn.Linear(model.fc.in_features, num_classes)  # Linear layer for classification
    )

    return model


def evaluate(model, loader, device):
    """
    Evaluate the model on a given dataset loader.
    """
    model.eval()  # Set model to evaluation mode

    correct = 0
    total = 0
    loss_total = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # Disable gradient computation
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            preds = torch.argmax(outputs, dim=1)  # Get predicted classes

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss_total += loss.item()

    accuracy = correct / total
    loss = loss_total / len(loader)

    return loss, accuracy


def train(model, train_loader, val_loader, device, epochs, lr, patience):
    """
    Train the model with early stopping based on validation accuracy.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0
    patience_counter = 0

    model.to(device)

    for epoch in range(epochs):
        model.train()  # Set model to training mode

        train_loss = 0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = train_loss / len(train_loader)

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

        # Check for improvement
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0

            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)

            # Save model checkpoint
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_loader.dataset.dataset.classes
            }, MODEL_PATH)

            print("New best model saved!")

        else:
            patience_counter += 1

            if patience_counter >= patience:

                print("Early stopping triggered")
                break


def main():
    """
    Main function to parse command-line arguments and run the training pipeline.
    """
    parser = argparse.ArgumentParser(description="Train a vision model on image classification dataset")

    # command-line options with explanatory help text
    parser.add_argument("--data_dir", default="data/animals10/raw-img",
                        help="Directory containing image subfolders per class")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for both training and validation")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="Learning rate for the optimizer")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="Early-stopping patience in epochs")

    args = parser.parse_args()

    # Determine device
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # Prepare data loaders
    train_loader, val_loader, classes = get_dataloaders(
        args.data_dir,
        args.batch_size
    )

    # Build model
    model = build_model(len(classes))

    # Train the model
    train(
        model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        args.patience
    )


if __name__ == "__main__":
    main()