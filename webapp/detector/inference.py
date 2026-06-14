import cv2
from facenet_pytorch import MTCNN
import numpy as np
import torch
from PIL import Image
import sys
import albumentations as A
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from src.model import get_model

model, device = get_model()
mtcnn = MTCNN(image_size=224, device=device, margin=20, thresholds=[0.6,0.7,0.7], keep_all=True)
checkpoint = torch.load("../models/efficientnet_b0/results/best__seed-3_e-10_augment-on_sampler-off_loss-bce_pw-0.15736747005.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    ),
    A.ToTensorV2()
])
num_frames = 16

def extract_faces(video):        
        cap=cv2.VideoCapture(str(video))
        
        if not cap.isOpened():
            return None
                      
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <=0:
            cap.release()
            return None
        indices = set(np.linspace(0, total_frames-1, num=min(total_frames,num_frames), dtype=int))
                
        current= 0
        frames = []
                
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                    
            if current in indices:
                frames.append(frame)
                    
            current+=1
                
        cap.release()
        all_faces=[]
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
                        
            with torch.inference_mode():
                faces, probs = mtcnn(pil_frame, return_prob=True)
                        
            if faces is not None:
                best_idx = np.argmax(probs)
                            
                face = faces[best_idx] 
                face = ((face.permute(1,2,0).cpu().numpy()+1)/2*255).astype("uint8")
                
                if transform is not None:
                    augmented = transform(image = face)
                    face = augmented['image']
                
                all_faces.append(face)

        if len(all_faces) == 0: 
            return {
            "faces": None,
            "frames_analyzed": len(frames),
            "faces_detected": len(all_faces),
            }
                    
        return {
            "faces": torch.stack(all_faces),
            "frames_analyzed": len(frames),
            "faces_detected": len(all_faces),
            }

def predict_video(video):
    threshold = 0.9993
    extracted = extract_faces(video=video)
    if extracted["faces"] is None:
        return {
            "probability": None,
            "prediction": "No faces detected",
            "frames_analyzed": extracted["frames_analyzed"],
            "faces_detected": extracted["faces_detected"],
            "threshold": threshold
        }

    faces = extracted["faces"]
    

    with torch.inference_mode():
        logits = model(faces.to(device, memory_format=torch.channels_last)).flatten()
        calculated_logit=torch.topk(logits, k=min(5,logits.numel())).values.mean()
        prob = torch.sigmoid(calculated_logit)
        prediction = "Fake" if prob.item() > threshold else "Real"
    
    return {
        "probability": prob.item(),
        "prediction": prediction,
        "frames_analyzed": extracted["frames_analyzed"],
        "faces_detected": extracted["faces_detected"],
        "threshold": threshold
    }    


