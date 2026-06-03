import torch
from pathlib import Path
from tqdm import tqdm
import logging
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
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--num-frames", type=int, default=16, required=False, help="Number of frames to process")
    parser.add_argument("--aggregation", type=str, default="topk", choices=["mean", "median", "topk", "max"], required=False, help="Choose method of aggregation")    
    
    args = parser.parse_args()
    
    compress = args.compress
    num_frames = args.num_frames
    aggregation = args.aggregation
    
    videos_dir = Path(f"../data/processed/celeb-df-v2")
    log_dir = Path("../models/efficientnet_b0/logs/celeb-df-v2_logs/")
    log_file = (
    "test_"
    f"{aggregation}_frames-{num_frames}"
    f"{'_compress' if compress else ''}.log"
)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s , %(message)s",
        handlers=[
                logging.FileHandler(log_dir / log_file),
                logging.StreamHandler()]
    )

    logger = logging.getLogger(__name__)
    
    logger.info(f"Aggregation: {aggregation}")
    
    if(compress):
        logger.info("Compression: on")
        eval_transforms = A.Compose([
        A.Resize(224,224),
        A.ImageCompression(quality_range=(20,90)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ToTensorV2(),
    ])
    else:
        logger.info("Compression: off")
        eval_transforms = A.Compose([
                          A.Resize(224,224),
                          A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          A.ToTensorV2(),
        ])
    
        logger.info("Testing...")
        eval_dataset = VideoDataset((videos_dir / "test"), transform=eval_transforms, num_frames=num_frames)
    
    
    model, device = get_model()
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    logger.info(f"Loading checkpoint {args.checkpoint}")    
    logger.info(f"Model running on {device}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = torch.compile(model)
    model.eval()
    
    all_probs =[]
    labels = []
    skipped = 0
    with torch.inference_mode():
        for i  in tqdm(range(len(eval_dataset)), desc="Testing"):
            faces,label = eval_dataset[i]
            if faces is None:
                skipped+=1
                continue
            faces = faces.to(device, non_blocking = True)
            faces = faces.to(memory_format = torch.channels_last)
                        
            logits = model(faces).flatten()           
            if aggregation=="mean":
                calculated_logit = logits.mean()
            elif aggregation =="max":
                calculated_logit = logits.max()
            elif aggregation=="median":
                calculated_logit = torch.median(logits)
            elif aggregation == "topk":
                k = min(5, logits.numel())
                if i==0:
                    logger.info(f"k={k}")
                calculated_logit = torch.topk(logits, k).values.mean()
            else:
                pass
            
            calculated_prob = torch.sigmoid(calculated_logit)
            labels.append(label)
            all_probs.append(calculated_prob.item())
    
    
    all_probs=np.array(all_probs)
    labels = np.array(labels)
    threshold = 0.9993
    all_preds = (all_probs > threshold).astype(float)
        
    auc = roc_auc_score(labels, all_probs)
    balanced_acc_score = balanced_accuracy_score(labels, all_preds)
    confusion_mat = confusion_matrix(labels, all_preds)
    classification_repo = classification_report(labels, all_preds, target_names=["real", "fake"])
        
    logger.info("Confusion matrix: \n %s", confusion_mat)
    logger.info("Classification report: \n %s", classification_repo)
    logger.info(      
                f"SUMMARY \n"
                f"Aggregation: {aggregation} \n"
                f"Frames sampled: {num_frames} \n"
                f"Compression: {'on' if compress else 'off'} \n"
                f"Threshold {threshold:.4f} \n"
                f"Balanced accuracy: {balanced_acc_score:.4f} \n"
                f"AUC: {auc:.4f} \n"
                f"Videos skipped: {skipped} \n"
                )
    
