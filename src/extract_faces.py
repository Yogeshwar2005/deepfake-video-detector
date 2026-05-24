import torch
import cv2
from facenet_pytorch import MTCNN
from pathlib import Path
from tqdm import tqdm
from PIL import Image

skip=10         # every 10th frame gets processed
max_faces=30    # 30 faces per video

def get_split(video_num):
    if video_num < 720:
        return "train"
    elif video_num <860:
        return "val"
    else:
        return "test"

def save_faces(videos, label, mtcnn, manipulation):
    for video in tqdm(videos, desc=f"Processing {manipulation}"):
        
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
                if not ret or saved_count >=max_faces:
                    break 
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)
                
                with torch.inference_mode():
                    faces, probs = mtcnn(pil_frame, return_prob=True)
                
                if faces is not None:
                    probs = list(probs)
                    best_idx = probs.index(max(probs))
                    
                    face = faces[best_idx] 
                    face_np = ((face.permute(1,2,0).cpu().numpy()+1)/2*255).astype("uint8")
                    face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
                    
                    if label == "fake":
                        name = f"{manipulation}_{Path(video.name).stem}_{saved_count}.jpg"
                    else:
                        name = f"{Path(video.name).stem}_{saved_count}.jpg"
                        
                    success = cv2.imwrite(f"../data/processed/{split}/{label}/{name}", face_bgr)
                    if success:
                        saved_count+=1
                
                for _ in range(skip-1):
                    cap.grab()        
        except Exception as e:
            print(f"Error processing {video.name}: {e}")
        finally:
            if cap is not None:
                cap.release()
           
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device}")
    
    mtcnn = MTCNN(image_size=224, device=device, margin=20, thresholds=[0.6,0.7,0.7], keep_all=True)
    
    raw_dir = Path("../data/raw")
    real_videos = list((raw_dir / "original" ).glob("*.mp4"))
    deepfakes_videos = list((raw_dir / "Deepfakes").glob("*.mp4"))
    deepfakedetection_videos = list((raw_dir / "DeepFakeDetection").glob("*.mp4"))
    face2face_videos = list((raw_dir / "Face2Face" ).glob("*.mp4"))
    faceshifter_videos = list((raw_dir / "FaceShifter" ).glob("*.mp4"))
    faceswap_videos = list((raw_dir / "FaceSwap" ).glob("*.mp4"))
    neuraltextures_videos = list((raw_dir / "NeuralTextures" ).glob("*.mp4"))

    save_faces(deepfakedetection_videos, "fake", mtcnn,"deepfakedetection")
    save_faces(face2face_videos, "fake", mtcnn, "face2face")
    save_faces(faceshifter_videos, "fake", mtcnn, "faceshifter")
    save_faces(faceswap_videos, "fake", mtcnn, "faceswap")
    save_faces(neuraltextures_videos, "fake", mtcnn, "neuraltextures")
    save_faces(deepfakes_videos, "fake", mtcnn, "deepfakes")
    
    save_faces(real_videos, "real", mtcnn, "real" )
           
            
            
    