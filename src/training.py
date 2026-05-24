import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append("../src/")
from model import get_model
from dataset import DeepfakeDataset


if __name__ == "__main__":
    train_transforms= transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224,224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]
                                    )
    eval_transforms= transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])]
                                        )
    

    train_dataset= DeepfakeDataset("../data/processed/train/", transform=train_transforms)
    val_dataset = DeepfakeDataset("../data/processed/val/", transform= eval_transforms)

    train_loader= DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader= DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model, device = get_model()
    print(f"Model running on {device}")

    criterion= nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    EPOCHS = 10
    
    best_accuracy=0.0
    for epoch in range(EPOCHS):
        model.train()

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        with torch.no_grad():
            correct=0
            total=0
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                images = images.to(device)
                labels = labels.to(device)
                labels = labels.float().unsqueeze(1)
                outputs = model(images)
                predictions = (outputs>0).float()
                correct+= (predictions==labels).float().sum().item()
                total+= labels.shape[0]
                
            accuracy = correct / total
        if accuracy > best_accuracy:
            best_accuracy=accuracy
            torch.save(model.state_dict(), "../checkpoints/best_model.pth")
            print(f"New best model saved: {accuracy:.4f}")
            
        print(f"Epoch: {epoch+1}/{EPOCHS}, Loss:{loss.item():.4f}, Val accuracy: {accuracy:.4f}")