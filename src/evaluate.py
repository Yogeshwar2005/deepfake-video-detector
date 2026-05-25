import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import argparse
from pathlib import Path
from sklearn.metrics import(
    confusion_matrix, balanced_accuracy_score,
    classification_report
)


import sys
sys.path.append("../src/")
from model import get_model
from dataset import ImagesDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model")
    parser.add_argument("-t","--threshold",type = float, required=False, default=None, help="Threshold for predicting fake")
    args = parser.parse_args()
    

    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = ImagesDataset("../data/processed/test/",transform = eval_transforms)
    test_loader = DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=32)
    
    model, device  = get_model()
    print(f"model running on {device}")
    
    checkpoint = torch.load(Path(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded checkpoint:", args.checkpoint)
    
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = checkpoint["threshold"]
    
    print(f"Threshold: {threshold}")

    
    model.eval()
    all_preds=[]
    all_labels = []
    with torch.inference_mode():
        for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                predictions = (probs> threshold).float()
                
                all_preds.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

    
    print("Confusion matrix: ")
    print(confusion_matrix(all_labels, all_preds))
            
    print("Classification report: ")
    print(classification_report(all_labels, all_preds, target_names=["real", "fake"]))

    balanced_acc_score = balanced_accuracy_score(all_labels, all_preds)
    print("Balanced accuracy:",balanced_acc_score)