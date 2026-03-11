import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


MODEL_PATH = "models/vision_model.pth"  # Path to the trained model checkpoint


def load_vision_model():
    """
    Load trained image classification model.
    """

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    # Extract the list of animal classes from checkpoint
    classes = checkpoint["classes"]

    # Create ResNet18 model architecture (without pretrained weights)
    model = models.resnet18()

    # Replace the final fully connected layer to match our number of classes
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, len(classes))
    )

    # Load the trained model weights
    model.load_state_dict(checkpoint["model_state"])

    # Set model to evaluation mode (disables dropout, batch norm updates)
    model.eval()

    return model, classes


def predict_image(image_path, model, classes):
    """
    Predict class of an image.
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Standard ResNet18 input dimensions
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet dataset mean values
            std=[0.229, 0.224, 0.225]    # ImageNet dataset std values
        )
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    image = transform(image)  # Apply preprocessing transforms
    image = image.unsqueeze(0)  # Add batch dimension (1, 3, 224, 224)

    # Run inference without gradient computation
    with torch.no_grad():
        outputs = model(image) 
        probs = torch.softmax(outputs, dim=1)  
        confidence, pred = torch.max(probs, dim=1)  # Get max probability and index

    # Convert predicted index to class name
    return classes[pred.item()], confidence.item()