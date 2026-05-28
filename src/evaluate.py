import torch
import logging
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
    parser.add_argument("-l", "--load", type=str, required=True, help="Path to model")
    parser.add_argument("-t","--threshold",type = float, required=False, default=None, help="Threshold for predicting fake")
    parser.add_argument("-c","--compress",type = int, required=False, default=0, help="Whether to apply compression transformation or not")

    args = parser.parse_args()
    name = Path(args.load).stem
    
    log_dir=Path("../logs/test_logs")
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
    
    
    
    test_dataset = ImagesDataset("../data/processed/test/",transform = eval_transforms)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             num_workers=4,
                             batch_size=32,
                             pin_memory=True,
                             persistent_workers=True,
                             worker_init_fn=seed_worker,
                             generator=g)
    
    model, device  = get_model()

    print(f"Model running on {device}...")

    checkpoint = torch.load(Path(args.load), map_location=device, weights_only=False)
    print(f"Loading checkpoint: {args.load}...")
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print("Compiling model...")
    model = torch.compile(model)
    model.eval()
       
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = checkpoint["global_threshold"]
    
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
    auc = roc_auc_score(all_labels, all_probs)
    balanced_acc_score = balanced_accuracy_score(all_labels, all_preds)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    classification_repo = classification_report(all_labels, all_preds, target_names=["real", "fake"])
    
    
    logger.info("Confusion matrix: \n %s", confusion_mat)
    logger.info("Classification report: \n %s", classification_repo)
    logger.info(
                f"SUMMARY \n"
                f"Compression: {'on' if args.compress else 'off'} \n"
                f"Threshold {threshold:.4f} \n"
                f"Balanced accuracy: {balanced_acc_score:.4f} \n"
                f"AUC: {auc:.4f} \n"
                f"Mean fake probability: {np.mean(all_probs):.4f} \n"
                )
    