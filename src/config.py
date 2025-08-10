import torch
from dataclasses import dataclass


@dataclass
class Config:
    # Training
    num_epochs: int = 100
    batch_size: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4
    smooth: float = 1e-6
    dropout: float = 0.1
    random_state: int = 42

    # Image
    image_size: tuple = (565, 585)

    # Model
    in_channels: int = 3
    num_classes: int = 1

    # Metrics
    threshold: float = 0.5

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
