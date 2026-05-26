import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.metrics import(
    confusion_matrix, balanced_accuracy_score,
    classification_report, roc_auc_score
)
import albumentations as A
import numpy as np

import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", type=str, required=True, help="Path to model")
    parser.add_argument("-t","--threshold",type = float, required=False, default=None, help="Threshold for predicting fake")
    parser.add_argument("-c","--compress",type = int, required=False, default=0, help="Whether to apply compression transformation or not")

    args = parser.parse_args()
    

    
    if(args.compress == 1):
        print("Compression transformation applied")
        eval_transforms = A.Compose([
        A.Resize(224,224),
        A.ImageCompression(quality_range=(20,90)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])
        
    else:
        print("No compression transformation applied")
        eval_transforms = A.Compose([
                          A.Resize(224,224),
                          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          A.ToTensorV2(),
        ])
    
    
    
    test_dataset = ImagesDataset("../data/processed/test/",transform = eval_transforms)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=32, pin_memory=True, persistent_workers=True)
    
    model, device  = get_model()

    print(f"Model running on {device}...")


    checkpoint = torch.load(Path(args.load), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded checkpoint:", args.load)
    
    print("Compiling model...")
    model = torch.compile(model)
    model.eval()
       
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = checkpoint["global_threshold"]
    
    print(f"Threshold: {threshold:.4f}")

    all_labels = []
    all_probs=[]
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device, non_blocking=True)
                images =images.to(memory_format = torch.channels_last)
                
                labels = labels.float().unsqueeze(1).to(device, non_blocking=True)
                
                outputs = model(images)
                
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())
    all_preds = (np.array(all_probs) > threshold).astype(float)

    print(f"Mean fake probability: {np.mean(all_probs):.4f}")
    print(f"Min probability: {np.min(all_probs):.4f}")
    print(f"Max probability: {np.max(all_probs):.4f}")
    
    print("Confusion matrix: ")
    print(confusion_matrix(all_labels, all_preds))
            
    print("Classification report: ")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))

    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC: {auc}")
    
    balanced_acc_score = balanced_accuracy_score(all_labels, all_preds)
    print(f"Balanced accuracy: {balanced_acc_score:.4f}")