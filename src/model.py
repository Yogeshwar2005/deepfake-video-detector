import torch
import torch.nn as nn
import torchvision.models as models

def get_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(in_features=1280,out_features=1)
    device= "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model = model.to(memory_format=torch.channels_last)
    return model, device
    