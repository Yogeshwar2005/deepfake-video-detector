import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append("../src/")
from model import get_model
from dataset import DeepfakeDataset

if __name__ == "__main__":
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = DeepfakeDataset("../data/processed/test/",transform = eval_transforms)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=32)
    
    model, device  = get_model()
    print(f"model running on {device}")
    
    model.load_state_dict(torch.load("../checkpoints/model_epoch10.pth", map_location=device))
    model.eval()
    
    correct=0
    total=0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                predictions =(outputs > 0).float()
                correct+= (predictions == labels).float().sum().item()
                total += labels.shape[0]
    accuracy = correct / total
    print("Accuracy:", accuracy)