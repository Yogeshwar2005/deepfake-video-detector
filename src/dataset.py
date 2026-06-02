import torch
from torch.utils.data import Dataset
import cv2
from pathlib import Path
from facenet_pytorch import MTCNN
import numpy as np
from PIL import Image

class ImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = list(Path(root_dir).rglob("*.jpg"))
        self.labels = [0 if image.parent.name=="real" else 1 for image in self.images]
        self.transform=transform
    def __len__(self):
        return len(self.images)
    def __getitem__(self,idx):
        image = cv2.imread(str(self.images[idx]))
        if image is None:
            print(f"failed to load image: {self.images[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, self.labels[idx]



class VideoDataset(Dataset):
    def __init__(self, root_dir, transform = None, num_frames = 30):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mtcnn = MTCNN(image_size=224, device=self.device, margin=20, thresholds=[0.6,0.7,0.7], keep_all=True)
        self.videos = list(Path(root_dir).rglob("*.mp4"))  
        self.labels = [0 if video.parent.name == "real" else 1 for video in self.videos]
        self.transform = transform
        self.num_frames = num_frames
    
    def __len__(self):
        return len(self.videos)
    
    def extract_faces(self,video):        
        cap=cv2.VideoCapture(str(video))
                                
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <=0:
            cap.release()
            return None
        indices = set(np.linspace(0, total_frames-1, num=min(total_frames,self.num_frames), dtype=int))
                
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
                faces, probs = self.mtcnn(pil_frame, return_prob=True)
                        
            if faces is not None:
                best_idx = np.argmax(probs)
                            
                face = faces[best_idx] 
                face = ((face.permute(1,2,0).cpu().numpy()+1)/2*255).astype("uint8")
                
                if self.transform is not None:
                    augmented = self.transform(image = face)
                    face = augmented['image']
                
                all_faces.append(face)

        if len(all_faces) == 0:
            return None
        return torch.stack(all_faces)
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        faces = self.extract_faces(video)
        
        return faces, self.labels[idx]
    
    