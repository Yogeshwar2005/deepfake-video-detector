import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from pathlib import Path


import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model")
    args = parser.parse_args()
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImagesDataset("../data/processed/test/",transform = eval_transforms)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=32)
    
    model, device  = get_model()
    print(f"model running on {device}")
    
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    correct=0
    total=0
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                predictions =(outputs > 0).float()
                correct+= (predictions == labels).float().sum().item()
                total += labels.shape[0]
    accuracy = correct / total
    print("Accuracy:", accuracy)