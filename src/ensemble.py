import torch
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.metrics import(
    confusion_matrix, 
    classification_report, roc_auc_score, roc_curve,
    balanced_accuracy_score
)
import albumentations as A
import numpy as np
import random
import os

import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset

def seed_everything(seed=42):
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
    
    SEED=42
    seed_everything(SEED)
    print("Seed:", SEED)
    
    g=torch.Generator()
    g.manual_seed(SEED)
    
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument("-t","--threshold",type = float, required=False, default=None, help="Threshold for predicting fake")
    parser.add_argument("-c","--compress",type = int, required=False, default=0, help="Whether to apply compression transformation or not")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument("--logic-gated", action="store_true")
    

    args = parser.parse_args()
    
    path_1="../models/efficientnet_b0/results/best__seed-3_e-10_augment-on_sampler-off_loss-bce_pw-0.15736747005.pth"
    path_2="../models/efficientnet_b0/checkpoints/best__seed-2_e-10_augment-on_sampler-off_loss-focal_alpha-0.15736747005_gamma-2.0.pth"
    validate = args.validate
    test = args.test
    
    name = f"{Path(path_1).stem}+{Path(path_2).stem}"
        
    log_dir=Path("../models/efficientnet_b0/logs/ensemble_logs")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s , %(message)s",
        handlers=[
                logging.FileHandler(log_dir / f"{name}.log"),
                logging.StreamHandler()]
    )

    logger = logging.getLogger(__name__)
    
    if(args.compress == 1):
        print("Compression: on")
        eval_transforms = A.Compose([
        A.Resize(224,224),
        A.ImageCompression(quality_range=(20,90)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])
        
    else:
        print("Compression: off")
        eval_transforms = A.Compose([
                          A.Resize(224,224),
                          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          A.ToTensorV2(),
        ])
    
    
    if test:
        eval_dataset = ImagesDataset("../data/processed/test/",transform = eval_transforms)
        logger.info("Testing...")
    elif validate:
        eval_dataset = ImagesDataset("../data/processed/val/",transform = eval_transforms)
        logger.info("Validating...")
    
    eval_loader = DataLoader(eval_dataset,
                             shuffle=False,
                             num_workers=4,
                             batch_size=32,
                             pin_memory=True,
                             persistent_workers=True,
                             worker_init_fn=seed_worker,
                             generator=g)
    
    model_1, device_1  = get_model()
    model_2, device_2 = get_model()
    
    if device_1 != device_2:
        print("Error: Models running on different devices")
        exit()

    device = device_1
    print(f"Model_1 running on {device}...")
    print(f"Model_2 running on {device}...")

    checkpoint_1 = torch.load(Path(path_1), map_location=device_1, weights_only=False)
    
    print(f"Loading checkpoint: {path_1}...")
    model_1.load_state_dict(checkpoint_1["model_state_dict"])
    
    checkpoint_2 = torch.load(Path(path_2), map_location=device_2, weights_only=False)
    print(f"Loading checkpoint: {path_2}...")
    model_2.load_state_dict(checkpoint_2["model_state_dict"])  
     
    print("Compiling models...")
    model_1 = torch.compile(model_1)
    model_2 = torch.compile(model_2)
    model_1.eval()
    model_2.eval()
    
    
    all_labels=[]
    all_probs_1=[]
    all_probs_2=[]
    probs_1 = []
    probs_2 = []
    all_probs=[]
    with torch.inference_mode():
        for images, labels in tqdm(eval_loader, desc=f"{'Validating' if validate else 'Testing'} "):            
            all_labels.extend(labels.cpu().numpy().flatten())                

            images = images.to(device, non_blocking=True)   
            images = images.to(memory_format = torch.channels_last)
                                        
            outputs_1 = model_1(images)
            outputs_2 = model_2(images)
                
            probs_1 = torch.sigmoid(outputs_1)
            probs_2 = torch.sigmoid(outputs_2)
            
            all_probs_1.extend(probs_1.cpu().numpy().flatten())
            all_probs_2.extend(probs_2.cpu().numpy().flatten())                    
    
    
    if args.logic_gated:
        
        all_probs = [all_probs_1[i] if (all_probs_1[i] > 0.9 or all_probs_1[i] < 0.1) else all_probs_2[i] for i in range(len(all_probs_1))]

        if validate:
            fpr,tpr,thresholds = roc_curve(all_labels, all_probs)        
            valid = np.isfinite(thresholds)
            thresholds = thresholds[valid]
                        
            balanced_acc_scores = (tpr + (1-fpr)) / 2
            balanced_acc_scores = balanced_acc_scores[valid]
                        
            best_idx = balanced_acc_scores.argmax()
            best_threshold = thresholds[best_idx]
            best_balanced_acc_score = balanced_acc_scores[best_idx]
    
        auc = roc_auc_score(all_labels, all_probs)
        all_preds = (np.array(all_probs) > (best_threshold if validate else 0.9547)).astype(float)
        confusion_mat = confusion_matrix(all_labels, all_preds)
        classification_repo = classification_report(all_labels, all_preds, target_names=["real", "fake"])
        
        logger.info("Confusion matrix: \n %s", confusion_mat)
        logger.info("Classification report: \n %s", classification_repo)
        logger.info(
                        f"SUMMARY \n"
                        f"Compression: {'on' if args.compress else 'off'} \n"
                        f"Threshold {0.9547 :.4f} \n"
                        f"Balanced accuracy: {balanced_accuracy_score(all_labels, all_preds):.4f} \n"
                        f"AUC: {auc:.4f} \n"
                        f"Mean probability: {np.mean(all_probs):.4f} \n"
                        f"Standard deviation: {np.std(all_probs):.4f} \n"
                    )
        
    
    if args.weighted:
        if validate:
            best_auc = 0.0 
            best_w1=0.0
            best_w2=0.0
            all_probs_1 = np.array(all_probs_1)                  
            all_probs_2 = np.array(all_probs_2)  
            all_labels=np.array(all_labels)                
            for w1 in np.arange(0.01,1,0.01):
                w2 = 1 - w1
                all_probs = w1*all_probs_1 + w2*all_probs_2   
                auc =  roc_auc_score(all_labels, all_probs)
                if auc > best_auc:
                    best_auc = auc
                    best_w1=w1
                    best_w2=w2
                    best_probs = all_probs
                    
            fpr,tpr,thresholds = roc_curve(all_labels, best_probs)        
            valid = np.isfinite(thresholds)
            thresholds = thresholds[valid]
                        
            balanced_acc_scores = (tpr + (1-fpr)) / 2
            balanced_acc_scores = balanced_acc_scores[valid]
                        
            best_idx = balanced_acc_scores.argmax()
            best_threshold = thresholds[best_idx]
            best_balanced_acc_score = balanced_acc_scores[best_idx]   
            best_preds = (np.array(best_probs) > best_threshold).astype(float)
            confusion_mat = confusion_matrix(all_labels, best_preds)
            classification_repo = classification_report(all_labels, best_preds, target_names=["real", "fake"])
            
            
            logger.info("Confusion matrix: \n %s", confusion_mat)
            logger.info("Classification report: \n %s", classification_repo)
            logger.info(
                        f"SUMMARY \n"
                        f"Compression: {'on' if args.compress else 'off'} \n"
                        f"Threshold {best_threshold:.4f} \n"
                        f"Best Weight 1 (Model 1): {best_w1:.2f} \n"     
                        f"Best Weight 2 (Model 2): {best_w2:.2f} \n"
                        f"Balanced accuracy: {best_balanced_acc_score:.4f} \n"
                        f"AUC: {best_auc:.4f} \n"
                        f"Mean probability: {np.mean(best_probs):.4f} \n"
                        f"Standard deviation: {np.std(best_probs):.4f} \n"
                        )
        
        if test: 
            w1 = 0.93
            w2 = 0.07
            threshold = 0.9360
            all_probs_1 = np.array(all_probs_1)
            all_probs_2 = np.array(all_probs_2)
            all_probs = w1*all_probs_1 + w2*all_probs_2
            all_preds = (np.array(all_probs) > threshold).astype(float)
            confusion_mat = confusion_matrix(all_labels, all_preds)
            classification_repo = classification_report(all_labels, all_preds, target_names=["real", "fake"])
            balanced_acc_score = balanced_accuracy_score(y_true=all_labels, y_pred=all_preds)
            auc = roc_auc_score(all_labels, all_probs)
            
            logger.info("Confusion matrix: \n %s", confusion_mat)
            logger.info("Classification report: \n %s", classification_repo)
            logger.info(
                        f"SUMMARY \n"
                        f"Compression: {'on' if args.compress else 'off'} \n"
                        f"Threshold {threshold:.4f} \n"
                        f"Weight 1 (Model 1): {w1:.2f} \n"     
                        f"Weight 2 (Model 2): {w2:.2f} \n"
                        f"Balanced accuracy: {balanced_acc_score:.4f} \n"
                        f"AUC: {auc:.4f} \n"
                        f"Mean probability: {np.mean(all_probs):.4f} \n"
                        f"Standard deviation: {np.std(all_probs):.4f} \n"
                        )
        

    
    