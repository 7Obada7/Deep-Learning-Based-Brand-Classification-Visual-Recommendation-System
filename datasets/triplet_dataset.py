import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os
from random import choice, sample
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root=data_dir)
        self.class_to_idx = self.dataset.class_to_idx  # {'audi': 0, 'bmw': 1, ...}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}  # {0: 'audi', 1: 'bmw', ...}
        self.classes = list(self.class_to_idx.keys())  # ['audi', 'bmw', 'honda', ...]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor_image, anchor_label = self.dataset[idx]
        
        anchor_class = self.idx_to_class[anchor_label]  # Convert integer -> class name
        
        # Positive sample (same class)
        positive_image_path = self._get_image_from_class(anchor_class)
        positive_image = self._load_image(positive_image_path)
        
        # Negative sample (different class)
        negative_class = choice([cls for cls in self.classes if cls != anchor_class])
        negative_image_path = self._get_image_from_class(negative_class)
        negative_image = self._load_image(negative_image_path)
        
        # Apply transformations
        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)
        
        return anchor_image, positive_image, negative_image

    def _get_image_from_class(self, class_name):
        class_path = os.path.join(self.data_dir, class_name)
        img_name = sample(os.listdir(class_path), 1)[0]
        return os.path.join(class_path, img_name)

    def _load_image(self, image_path):
        return Image.open(image_path).convert('RGB')  # Always convert to RGB
