# IMPORTS
import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
from torch.utils.data import (DataLoader,WeightedRandomSampler)
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import albumentations as A
import os
import random

import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument("--epochs", type=int, default=10, required=False, help="Total number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, required=False, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, required=False, help="Number of workers")
    parser.add_argument("--seed",type=int, default=42, required=False, help="Select seed (default=42)")
    
    parser.add_argument("--load", type=str,default=None,required=False, help="Path to checkpoint to resume training")
    
    parser.add_argument("--augment", action="store_true", help="Toggle augmentation")
    parser.add_argument("--sampler", action="store_true" , help="Toggle sampler")
    
    parser.add_argument("--loss", type=str, default="bce",choices=["bce", "focal", "topk"], required=False, help="Select loss function")
    parser.add_argument("--pos-weight", type=float, default=0.15736747005, required=False, help="Value of pos_weight for bce loss")
    parser.add_argument("--alpha", type=float,default=0.15736747005, required=False, help="Value of alpha for focal loss")
    parser.add_argument("--gamma", type=float,default=2.0, required=False, help="Value of gamma for focal loss")
    parser.add_argument("--k", type=float, default=0.8,required=False, help="Value of k for topk")
    

    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS=args.num_workers
    SEED = args.seed
    LOAD = args.load 
    AUGMENT = args.augment
    SAMPLER = args.sampler
    LOSS = args.loss
    ALPHA = args.alpha
    GAMMA = args.gamma
    POS_WEIGHT=args.pos_weight
    K = args.k

    seed_everything(SEED)
    print("Seed:", SEED)
    g=torch.Generator()
    g.manual_seed(SEED)
    
    torch.set_float32_matmul_precision('high')

    print(f"Running {EPOCHS} epochs with {BATCH_SIZE} batch size and {NUM_WORKERS} workers")
    print(f"Augmentation: {'on' if AUGMENT else 'off'}")
    print(f"Sampler: {'on' if SAMPLER else 'off'}")
    print(f"Loss function: {LOSS}")    
    
    if AUGMENT:
        print("Using augmentations...")
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
    
    if SAMPLER:
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
        train_loader= DataLoader(train_dataset, batch_size=BATCH_SIZE, generator=g, worker_init_fn=seed_worker,sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, generator=g, worker_init_fn=seed_worker, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    val_loader= DataLoader(val_dataset, batch_size=BATCH_SIZE,shuffle=False, num_workers=NUM_WORKERS,pin_memory=True, persistent_workers=True)

    model, device = get_model()
    print(f"Model running on {device}...")

    
    if (LOSS == "bce"):
        print(f"Using pos_weight={POS_WEIGHT}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]).to(device))
    elif(LOSS=="topk"):
        print(f"Using k={K}")
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    else:
        print(f"Using alpha={ALPHA} and gamma={GAMMA}")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS)

    start_epoch = 0
    global_balanced_acc_score=0.0
    global_threshold=0.5
    global_auc=0.0
    
    if LOAD is not None:
        checkpoint = torch.load(args.load, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        global_balanced_acc_score = checkpoint["global_balanced_acc_score"]
        global_threshold = checkpoint["global_threshold"]
        global_auc = checkpoint["global_auc"]
        print(f"Resuming training from epoch {start_epoch}...")
    
    print("Compiling model...")
    model = torch.compile(model)
    
    name = (
        f"seed-{SEED}"
        f"_e-{EPOCHS}"
        f"_augment-{'on' if AUGMENT else 'off'}"
        f"_sampler-{'on' if SAMPLER else 'off'}"
        f"_loss-{LOSS}"
        )
    
    if LOSS == "bce":
        name+=f"_pw-{POS_WEIGHT}"
    elif LOSS == "focal":
        name+=f"_alpha-{ALPHA}_gamma-{GAMMA}"
    else:
        name+=f"_k-{K}"
       
     
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        running_loss=0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training"):
            images = images.to(device, non_blocking=True)
            images = images.to(memory_format=torch.channels_last)
            
            labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = model(images)

            if(LOSS == "focal"):
                loss = torchvision.ops.sigmoid_focal_loss(
                inputs= outputs,
                targets=labels,
                alpha=ALPHA,
                gamma=GAMMA,
                reduction="mean"
            )
            elif(LOSS=="bce"):
                loss=criterion(outputs, labels)
            else:
                losses = criterion(outputs, labels).squeeze(1)
                k = max(1,int(K * len(labels)))
                hard_losses, _ = torch.topk(losses, k)
                loss = hard_losses.mean()
                
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
            
            valid = np.isfinite(thresholds)
            thresholds = thresholds[valid]
            
            balanced_acc_scores = (tpr + (1-fpr)) / 2
            balanced_acc_scores = balanced_acc_scores[valid]
            
            best_idx = balanced_acc_scores.argmax()
            epoch_threshold = thresholds[best_idx]
            epoch_balanced_acc_score = balanced_acc_scores[best_idx]
            
            
        all_preds = (all_probs > epoch_threshold).astype(float)
        epoch_auc = roc_auc_score(all_labels, all_probs)
            
        is_best = epoch_auc > global_auc
        if is_best:
            global_auc = epoch_auc
            global_balanced_acc_score = epoch_balanced_acc_score
            global_threshold=epoch_threshold
            cp ={
                "epoch": epoch +1,
                "model_state_dict": model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_balanced_acc_score": global_balanced_acc_score,
                "global_threshold": global_threshold,
                "global_auc": global_auc
            }
            torch.save(cp,f"../checkpoints/best__{name}.pth")
            print(f"New best model saved with AUC: {global_auc:.4f} at best__{name}.pth")
        
        cp ={
                "epoch": epoch +1,
                "model_state_dict": model._orig_mod.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_balanced_acc_score": global_balanced_acc_score,
                "global_threshold": global_threshold,
                "global_auc": global_auc,
                "epoch_balanced_acc_score": epoch_balanced_acc_score,
                "epoch_threshold": epoch_threshold,
                "epoch_auc": epoch_auc
            }
        torch.save(cp,f"../checkpoints/latest__{name}.pth")
        
        lr = optimizer.param_groups[0]["lr"]
        confusion_mat = confusion_matrix(all_labels, all_preds)
        classification_repo = classification_report(all_labels, all_preds, target_names=["real", "fake"])
        
        print("Confusion matrix:")
        print(confusion_mat)
        print("Classification report:")
        print(classification_repo)
        print(
            f"Epoch: {epoch+1}/{EPOCHS} \n" 
            f"Loss:{avg_loss:.4f} \n" 
            f"lr: {lr:} \n"
            f"Balanced accuracy of epoch {epoch+1}: {epoch_balanced_acc_score:.4f} \n" 
            f"Balanced accuracy of best model: {global_balanced_acc_score:.4f} \n"
            f"AUC of epoch {epoch+1}: {epoch_auc:.4f} \n"
            f"AUC of best model: {global_auc:.4f} \n"
            f"Threshold of epoch {epoch+1}: {epoch_threshold:.4f} \n" 
            f"Threshold of best model: {global_threshold:.4f} \n"
            )
      
        
        scheduler.step()