import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import albumentations as A
import random
import os
import sys
import argparse
from sklearn.metrics import(
    confusion_matrix, balanced_accuracy_score,
    classification_report, roc_auc_score, roc_curve
)
sys.path.append("../src/")
from dataset import VideoDataset
from model import get_model

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
    
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint", type=str, required=False, default="../models/efficientnet_b0/results/best__seed-3_e-10_augment-on_sampler-off_loss-bce_pw-0.15736747005.pth", help="Select checkpoint")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--compress", action="store_true")
    
    args = parser.parse_args()
    
    validate = args.validate
    test = args.test
    compress = args.compress
    
    videos_dir = Path(f"../data/processed/videos/")
    
    if(compress):
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
        print("Testing...")
        eval_dataset = VideoDataset((videos_dir / "test"), transform=eval_transforms, num_frames=30)
    elif validate:
        print("Validating...")
        eval_dataset=VideoDataset((videos_dir / "val"), transform=eval_transforms, num_frames=30)
    else:
        print("Error: Use --validate or --test")
        exit()
    
    
    model, device = get_model()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    print(f"Loading checkpoint {args.checkpoint}")    
    print("Model running on", device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = torch.compile(model)
    model.eval()
    
    all_probs =[]
    labels = []
    skipped = 0
    with torch.inference_mode():
        for i  in tqdm(range(len(eval_dataset)), desc=f"{'Validating' if validate else 'Testing'}"):
            faces,label = eval_dataset[i]
            if faces is None:
                skipped+=1
                continue
            faces = faces.to(device, non_blocking = True)
            faces = faces.to(memory_format = torch.channels_last)
                        
            logits = model(faces)
            probs = torch.sigmoid(logits)
            
            calculated_prob = probs.mean().item()
            labels.append(label)
            all_probs.append(calculated_prob)
    
    if validate:
        all_probs=np.array(all_probs)
        labels = np.array(labels)
        fpr,tpr, thresholds = roc_curve(labels, all_probs)
        
        valid = np.isfinite(thresholds)
        thresholds=thresholds[valid]
        
        balanced_acc_scores = (tpr+(1-fpr)) / 2
        balanced_acc_scores = balanced_acc_scores[valid]
        
        best_idx = balanced_acc_scores.argmax()
        threshold = thresholds[best_idx]
        balanced_acc_score = balanced_acc_scores[best_idx]
        
        all_preds = (all_probs > threshold).astype(float)
        
        auc = roc_auc_score(labels, all_probs)
        confusion_mat = confusion_matrix(labels, all_preds)
        classification_repo = classification_report(labels, all_preds, target_names=["real", "fake"])
    else:
        all_probs=np.array(all_probs)
        labels = np.array(labels)
        threshold = 0.6597
        all_preds = (all_probs > threshold).astype(float)
        
        auc = roc_auc_score(labels, all_probs)
        balanced_acc_score = balanced_accuracy_score(labels, all_preds)
        confusion_mat = confusion_matrix(labels, all_preds)
        classification_repo = classification_report(labels, all_preds, target_names=["real", "fake"])
        
    print("Confusion matrix: \n %s", confusion_mat)
    print("Classification report: \n %s", classification_repo)
    print(      
                f"SUMMARY \n"
                
                f"Compression: {'on' if compress else 'off'} \n"
                f"Threshold {threshold:.4f} \n"
                f"Balanced accuracy: {balanced_acc_score:.4f} \n"
                f"AUC: {auc:.4f} \n"
                f"Videos skipped: {skipped} \n"
                )
    
