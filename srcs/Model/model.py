from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


def build_model(num_classes: int) -> nn.Module:
    """Modify pre-trained ResNet-18 to classify our dataset."""
    # initialized with weights trained on the ImageNet dataset (1,000 classes).
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # the number of input features going into this layer
    in_features = model.fc.in_features
    # we change the nouber of class to match our purpose
    model.fc = nn.Linear(in_features, num_classes)
    return model
