import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
from pathlib import Path

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
            image = self.transform(image)
        return image, self.labels[idx]
        