from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torchvision.transforms as transforms

class MoEMIDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self._load_data()

    def _load_data(self):
        domains = ['ffhq', 'food101', 'places365']
        for label, domain in enumerate(domains):
            domain_dir = os.path.join(self.root_dir, 'raw', domain)
            for img_file in os.listdir(domain_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(domain_dir, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])