# IMPORTS

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import (DataLoader,WeightedRandomSampler)
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    balanced_accuracy_score,
    roc_curve,
    roc_auc_score
)
import albumentations as A


import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset


if __name__ == "__main__":

    torch.set_float32_matmul_precision('high')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("-e", "--epochs", type=int, default=10, required=False, help="Total number of epochs")
    parser.add_argument("-bs", "--batch-size", type=int, default=32, required=False, help="Batch size")
    parser.add_argument("-n", "--num-workers", type=int, default=4, required=False, help="Number of workers")
    parser.add_argument("-l", "--load", type=str,default=None,required=False, help="Path to checkpoint to resume training")
    parser.add_argument("-s","--sampler", type=int, default=0, required=False, help="Toggle sampler (1=on, 0=off)")
    parser.add_argument("-pw","--pos-weight", type=int, default=0, required=False, help="Toggle POS_WEIGHT (1=on, 0=off)")
    parser.add_argument("-a", "--augment",type=int, default=1, required=False, help="Toggle augmentation (1=on, 0=off)")

    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS=args.num_workers
    SAMPLER = args.sampler
    POS_WEIGHT=args.pos_weight
    AUGMENT = args.augment

    print(f"Running {EPOCHS} epochs with {BATCH_SIZE} batch size and {NUM_WORKERS} workers")
    print(f"Augmentation: {'on' if AUGMENT else 'off'}")
    print(f"Sampler: {'on' if SAMPLER else 'off'}")
    print(f"Pos weight: {'on' if POS_WEIGHT else 'off'}")    
    
    if AUGMENT == 1:
        train_transforms= A.Compose([
                                        A.Resize(224,224),
                                        A.HorizontalFlip(p=0.5),
                                        A.OneOf([
                                            A.ImageCompression(quality_range=(20,90), p=1.0),
                                            A.GaussNoise(std_range= (0.01,0.05  ), p=1.0),
                                            ], p=0.5),
                                        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
                                        A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                                        A.ToTensorV2()]
        )
    else:
        train_transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])
        
    eval_transforms = A.Compose([
        A.Resize(224,224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])
    

    train_dataset= ImagesDataset("../data/processed/train/", transform=train_transforms)
    val_dataset = ImagesDataset("../data/processed/val/", transform= eval_transforms)
    
    if(SAMPLER == 1):
        print("Using sampler...")
        targets = train_dataset.labels
        class_counts = np.bincount(targets)
        class_weights = 1.0/class_counts
        sample_weights = torch.DoubleTensor([class_weights[label] for label in targets])
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader= DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader= DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,pin_memory=True, persistent_workers=True)

    model, device = get_model()
    print(f"Model running on {device}...")

    
    if (POS_WEIGHT == 1): 
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.35]).to(device))
    else:
        criterion= nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)

    start_epoch = 0
    global_balanced_acc_score=0.0
    global_threshold=0.5
    
    if args.load is not None:
        checkpoint = torch.load(args.load, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_balanced_acc_score = checkpoint["global_balanced_acc_score"]
        global_threshold = checkpoint["global_threshold"]
        print(f"Resumed training from epoch {start_epoch}")
    
    print("Compiling model...")
    model = torch.compile(model)
    
    name = f"{timestamp}_e{EPOCHS}_bs{BATCH_SIZE}_aug{AUGMENT}_sampler{SAMPLER}_pw{POS_WEIGHT}"
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss=0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training"):
            images = images.to(device, non_blocking=True)
            images = images.to(memory_format=torch.channels_last)
            
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            
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
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Validating"):
                
                images = images.to(device)
                images = images.to(memory_format=torch.channels_last)

                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
                outputs = model(images)
                
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
                    
          
            all_probs = np.array(all_probs)
            fpr,tpr,thresholds = roc_curve(all_labels, all_probs)
            balanced_acc_scores = (tpr + (1-fpr)) / 2
            best_idx = balanced_acc_scores.argmax()
            epoch_threshold = thresholds[best_idx]
            epoch_balanced_acc_score = balanced_acc_scores[best_idx]
            
            
        all_preds = (all_probs > epoch_threshold).astype(float)
        

        print("Confusion matrix:")
        print(confusion_matrix(all_labels, all_preds))

        print("Classification report:")
        print(classification_report(
            all_labels,
            all_preds,
            target_names=["real", "fake"]
        ))
        
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC {auc}")
        
        
        is_best = epoch_balanced_acc_score > global_balanced_acc_score
        if is_best:
            global_balanced_acc_score=epoch_balanced_acc_score
            global_threshold=epoch_threshold
        
            cp ={
                "epoch": epoch +1,
                "model_state_dict": model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_balanced_acc_score": global_balanced_acc_score,
                "global_threshold": global_threshold
            }
            torch.save(cp,f"../checkpoints/{name}_best.pth")
            print(f"New best model saved with balanced accuracy: {global_balanced_acc_score:.4f} at {name}_best.pth")
        
        cp ={
                "epoch": epoch +1,
                "model_state_dict": model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_balanced_acc_score": global_balanced_acc_score,
                "epoch_balanced_acc_score": epoch_balanced_acc_score,
                "global_threshold": global_threshold,
                "epoch_threshold": epoch_threshold
            }
        torch.save(cp,f"../checkpoints/{name}_latest.pth")
        
        lr = optimizer.param_groups[0]["lr"]
        print(f"Threshold of epoch {epoch+1}: {epoch_threshold:.4f}")
        print(f"Threshold of best model: {global_threshold:.4f}")
        print(
            f"Epoch: {epoch+1}/{EPOCHS} \n" 
            f"Loss:{avg_loss:.4f} \n" 
            f"Balanced accuracy of epoch: {epoch_balanced_acc_score:.4f} \n" 
            f"Balanced accuracy of best model: {global_balanced_acc_score:.4f} \n"
            f"lr: {lr} \n"
            )
        
        scheduler.step()