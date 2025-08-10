import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import List
from .config import Config


norm = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


class RetinaDataset(Dataset):
    def __init__(
        self,
        image_paths: List[str],
        mask_paths: List[str],
        image_size=Config.image_size,
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=norm["mean"], std=norm["std"]),
            ]
        )

        self.mask_transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor()]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        image_tensor: torch.Tensor = self.image_transform(image)
        mask_tensor: torch.Tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return image_tensor, mask_tensor
