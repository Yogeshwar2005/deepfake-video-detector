import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime


import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset


if __name__ == "__main__":

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    parser=argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=10, required=False, help="Total number of epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=32, required=False, help="Batch size")
    parser.add_argument("-n", "--num_workers", type=int, default=4, required=False, help="Number of workers")
    parser.add_argument("-r", "--resume", type=str,default=None,required=False, help="Path to checkpoint to resume training")
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS=args.num_workers

    print(f"Running {EPOCHS} epochs with {BATCH_SIZE} batch size and {NUM_WORKERS} workers")
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
    

    train_dataset= ImagesDataset("../data/processed/train/", transform=train_transforms)
    val_dataset = ImagesDataset("../data/processed/val/", transform= eval_transforms)

    train_loader= DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader= DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model, device = get_model()
    print(f"Model running on {device}")

    criterion= nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    start_epoch = 0
    best_accuracy=0.0
    
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        
        print(f"Resumed training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss=0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train"):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        
        avg_loss = running_loss / len(train_loader)    
        model.eval()
        with torch.inference_mode():
            correct=0
            total=0
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                predictions = (outputs>0).float()
                correct+= (predictions==labels).float().sum().item()
                total+= labels.shape[0]
                
            accuracy = correct / total
        
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy=accuracy
        
        cp ={
                "epoch": epoch +1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_accuracy": best_accuracy
            }
        
        if is_best:
            torch.save(cp,"../checkpoints/best.pth")
            print(f"New best model saved: {accuracy:.4f}")
        
        
        torch.save(cp,"../checkpoints/latest.pth")
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch: {epoch+1}/{EPOCHS}, Loss:{avg_loss:.4f}, Val accuracy: {accuracy:.4f}, lr: {lr}")