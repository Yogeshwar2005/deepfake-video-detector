import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score
)


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

    criterion= nn.BCEWithLogitsLoss()# (pos_weight=torch.tensor([]).to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)


    start_epoch = 0
    global_best_balanced_acc_score=0.0
    global_best_threshold=0.5
    
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_best_balanced_acc_score = checkpoint["global_best_balanced_acc_score"]
        global_best_threshold = checkpoint["global_best_threshold"]
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
        
        all_probs = []
        all_labels = []
        with torch.inference_mode():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Val"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                    
            thresholds = np.arange(0.01, 1.0, 0.01)
            epoch_best_threshold = 0.5
            epoch_best_balanced_acc_score = 0.0
            all_probs = np.array(all_probs) 
            for t in thresholds:
                all_preds = (all_probs > t).astype(float)
                balanced_acc_score = balanced_accuracy_score(all_labels, all_preds)
                if (epoch_best_balanced_acc_score < balanced_acc_score):
                    epoch_best_threshold = t
                    epoch_best_balanced_acc_score = balanced_acc_score
        
        all_preds = (all_probs > epoch_best_threshold).astype(float)
        

        print("Confusion matrix:")
        print(confusion_matrix(all_labels, all_preds))

        print("Classification report:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=["real", "fake"]
        ))
                
        
        
        is_best = epoch_best_balanced_acc_score > global_best_balanced_acc_score
        if is_best:
            global_best_balanced_acc_score=epoch_best_balanced_acc_score
            global_best_threshold=epoch_best_threshold
        
            cp ={
                "epoch": epoch +1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_best_balanced_acc_score": global_best_balanced_acc_score,
                "global_best_threshold": global_best_threshold
            }
            torch.save(cp,"../checkpoints/best.pth")
            print(f"New best model saved: {global_best_balanced_acc_score:.4f}")
        
        cp ={
                "epoch": epoch +1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_best_balanced_acc_score": global_best_balanced_acc_score,
                "balanced_acc_score": epoch_best_balanced_acc_score,
                "global_best_threshold": global_best_threshold,
                "latest_threshold": epoch_best_threshold
            }
        torch.save(cp,"../checkpoints/latest.pth")
        lr = optimizer.param_groups[0]["lr"]
        print(f"Best threshold for epoch {epoch+1}: {epoch_best_threshold}")
        print(f"Best model threshold: {global_best_threshold}")
        print(
            f"Epoch: {epoch+1}/{EPOCHS}," 
            f"Loss:{avg_loss:.4f}," 
            f"Balanced accuracy: {epoch_best_balanced_acc_score:.4f}," 
            f"Best balanced accuracy: {global_best_balanced_acc_score},"
            f"lr: {lr}"
            )
    