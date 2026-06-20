import cv2
from facenet_pytorch import MTCNN
import numpy as np
import torch
from PIL import Image
import sys
import albumentations as A
from pathlib import Path
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


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
            return {
                "faces": None,
                "frames_analyzed": 0,
                "faces_detected": 0,
                "original_faces": [],
            }
                      
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <=0:
            cap.release()
            return {
                "faces": None,
                "frames_analyzed": 0,
                "faces_detected": 0,
                "original_faces": [],
            }        
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
        original_faces=[]
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(frame_rgb)
                        
            with torch.inference_mode():
                faces, probs = mtcnn(pil_frame, return_prob=True)
                        
            if faces is not None:
                best_idx = np.argmax(probs)
                            
                face = faces[best_idx] 
                face = ((face.permute(1,2,0).cpu().numpy()+1)/2*255).astype("uint8")
                
                original_faces.append(face.copy())
                
                if transform is not None:
                    augmented = transform(image = face)
                    face = augmented['image']
                
                all_faces.append(face)

        if len(all_faces) == 0: 
            return {
            "faces": None,
            "frames_analyzed": len(frames),
            "faces_detected": len(all_faces),
            "original_faces": [],
            }
                    
        return {
            "faces": torch.stack(all_faces),
            "frames_analyzed": len(frames),
            "faces_detected": len(all_faces),
            "original_faces": original_faces,
            }


def generate_gradcam(face_tensor, original_face, filename):
    model.eval()
    input_tensor = face_tensor.unsqueeze(0).to(device)
    
    target_layers = [model.features[-1]]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]
    
    original_face_float = original_face.astype(np.float32) / 255.0
    
    cam_image = show_cam_on_image(
        original_face_float,
        grayscale_cam,
        use_rgb=True
    )
    
    gradcam_dir = Path("media/gradcam")
    original_dir = Path("media/original")

    gradcam_save_path = gradcam_dir / filename
    original_save_path = original_dir / filename

    cv2.imwrite(
        str(gradcam_save_path),
        cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    )

    cv2.imwrite(
        str(original_save_path),
        cv2.cvtColor(original_face, cv2.COLOR_RGB2BGR)
    )

    return {
        "gradcam_path": f"gradcam/{filename}",
        "original_path": f"original/{filename}",
    }


def predict_video(video, result_id):
    threshold = 0.9993
    extracted = extract_faces(video=video)
    if extracted["faces"] is None:
        return {
            "probability": None,
            "prediction": "No faces detected",
            "frames_analyzed": extracted["frames_analyzed"],
            "faces_detected": extracted["faces_detected"],
            "threshold": threshold,
            "gradcam_paths": [],
            "orginal_paths": [],
        }

    faces = extracted["faces"]
    original_faces = extracted["original_faces"]
    

    with torch.inference_mode():
        logits = model(faces.to(device, memory_format=torch.channels_last)).flatten()
        calculated_logit=torch.topk(logits, k=min(5,logits.numel())).values.mean()
        prob = torch.sigmoid(calculated_logit)
        prediction = "Fake" if prob.item() > threshold else "Real"
    
    top_indices = torch.topk(
        logits,
        k=min(3, logits.numel())
    ).indices
    
    gradcam_results = []
    
    for rank, i in enumerate(top_indices):
        result = generate_gradcam(
            face_tensor=faces[i],
            original_face=original_faces[i],
            filename=f"{result_id}_{rank}.jpg"
        )
        gradcam_results.append(result)
    
    return {
        "probability": prob.item(),
        "prediction": prediction,
        "frames_analyzed": extracted["frames_analyzed"],
        "faces_detected": extracted["faces_detected"],
        "threshold": threshold,
        "gradcam_results":gradcam_results
    }    


