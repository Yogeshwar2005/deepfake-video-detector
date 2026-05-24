import torch
import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(in_features=1280,out_features=1)
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, device
    