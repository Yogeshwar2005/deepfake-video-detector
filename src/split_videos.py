import shutil
from pathlib import Path
from tqdm import tqdm


def get_split(video_num):
    if video_num < 720:
        return "train"
    elif video_num <860:
        return "val"
    else:
        return "test"
    
def split_videos(videos, label, manipulation):
    for video in tqdm(videos,desc=f"Splitting {manipulation}" ):
        if label == "real":
            video_num = int(Path(video).stem)
        else:
            video_num = int(Path(video).stem.split("_")[0])
        
        split = get_split(video_num=video_num)
        output_dir = Path(f"../data/processed/videos/{split}/{label}")
        
        video_name = f"{manipulation}_{Path(video).stem}.mp4"
        
        
        
        shutil.copy2(Path(video), (output_dir / video_name))


text_path= Path("../data/raw/celeb-df-v2/List_of_testing_videos.txt")
with open(text_path) as f:
    test_videos = {
        line.strip().split("/")[-1] for line in f
    }
    
    
    
def get_split_celeb(video_name):
    return "test" if video_name in test_videos else None

def split_videos_celeb(videos, label):
    for video in tqdm(videos,desc=f"Splitting celeb-df-v2" ):
        video_name = f"{Path(video).stem}.mp4"
        
        split = get_split_celeb(video_name=video_name)
        if not split:
            continue
        output_dir = Path(f"../data/processed/celeb-df-v2/{split}/{label}")
        
        shutil.copy2(Path(video), (output_dir / video_name))
    


if __name__ == "__main__":
    
    raw_dir = Path("../data/raw")
    real_videos = list((raw_dir / "original" ).glob("*.mp4"))
    deepfakes_videos = list((raw_dir / "Deepfakes").glob("*.mp4"))
    deepfakedetection_videos = list((raw_dir / "DeepFakeDetection").glob("*.mp4"))
    face2face_videos = list((raw_dir / "Face2Face" ).glob("*.mp4"))
    faceshifter_videos = list((raw_dir / "FaceShifter" ).glob("*.mp4"))
    faceswap_videos = list((raw_dir / "FaceSwap" ).glob("*.mp4"))
    neuraltextures_videos = list((raw_dir / "NeuralTextures" ).glob("*.mp4"))
    
    split_videos(real_videos, "real", manipulation="real")
    
    split_videos(deepfakes_videos, "fake", manipulation="deepfake")
    split_videos(deepfakedetection_videos, "fake", manipulation="deepfakedetection")
    split_videos(face2face_videos, "fake", manipulation="face2face")
    split_videos(faceshifter_videos, "fake", manipulation="faceshifter")
    split_videos(faceswap_videos, "fake", manipulation="faceswap")
    split_videos(neuraltextures_videos, "fake", manipulation="neuraltextures")

    raw_dir = Path("../data/raw/celeb-df-v2")
    celeb_real_videos = list((raw_dir / "Celeb-real").glob("*.mp4"))
    celeb_synthesis_videos = list((raw_dir / "Celeb-synthesis").glob("*.mp4"))
    youtube_real_videos = list((raw_dir / "YouTube-real").glob("*.mp4"))
    
    split_videos_celeb(celeb_real_videos, "real")
    split_videos_celeb(celeb_synthesis_videos, "fake")
    split_videos_celeb(youtube_real_videos, "real")
    
    print("real count:",len(list(Path("../data/processed/celeb-df-v2/test/real").glob("*.mp4"))))
    print("fake count:",len(list(Path("../data/processed/celeb-df-v2/test/fake").glob("*.mp4"))))

        
        