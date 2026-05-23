import torch
import cv2
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
mtcnn = MTCNN(image_size=224, device=device, margin=20, thresholds=[0.9,0.9,0.9])
raw_dir = Path("../data/raw")
real_videos = list((raw_dir / "original" ).glob("*.mp4"))
fake_videos = list((raw_dir / "Deepfakes").glob("*.mp4"))

def get_split(video_num):
    if video_num < 720:
        return "train"
    elif video_num <860:
        return "val"
    else:
        return "test"

def save_faces(videos, label):
    for video in tqdm(videos, desc=f"Processing {label}"):
        cap = None
        try:
            cap = cv2.VideoCapture(str(video)) 
            if label == "real":
                video_num = int(Path(video.name).stem)
            else:
                video_num = int(Path(video.name).stem.split("_")[0])
            split = get_split(video_num=video_num)
            saved_count=0
            while True:
                ret, frame = cap.read()
                if not ret or saved_count >=30:
                    break 
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                face = mtcnn(pil_frame)
                if face is not None:
                    face_np = ((face.permute(1,2,0).numpy()+1)/2*255).astype("uint8")
                    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    
                    name = f"{Path(video.name).stem}_{saved_count}.jpg"
                    
                    cv2.imwrite(f"../data/processed/{split}/{label}/{name}", face_bgr)
                    saved_count+=1
                for _ in range(9):
                    cap.grab()        
        except Exception as e:
            print(f"Error processing {video.name}: {e}")
        finally:
            if cap is not None:
                cap.release()
           
if __name__ == "__main__":
    save_faces(fake_videos,"fake")
    save_faces(real_videos,"real") 
           
            
            
    