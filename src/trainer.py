import torch.nn as nn
import torch.optim as optim
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from .config import Config
from typing import Dict, List, Tuple
import time


class UNetTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device = Config.device,
        weight_decay: float = Config.weight_decay,
    ):
        self.model = model.to(device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        self.weight_decay = weight_decay

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.lr,
            weight_decay=self.weight_decay,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            patience=5,
            factor=0.8,
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def dice_loss(self, pred, target, smooth=Config.smooth):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc="Training")

        for images, masks in progress_bar:
            images, masks = images.to(self.device), masks.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks) + self.dice_loss(outputs, masks)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return total_loss / num_batches

    def validate_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_dice = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, masks in self.val_loader:

                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)

                loss = self.criterion(outputs, masks) + self.dice_loss(outputs, masks)
                total_loss += loss.item()
                total_dice += 1 - self.dice_loss(outputs, masks).item()

        return total_loss / num_batches, total_dice / num_batches

    def train(
        self,
        num_epochs: int = Config.num_epochs,
    ) -> Dict[str, List[float]]:
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Training phase
            train_loss = self.train_epoch()

            # Validation phase
            val_loss, val_dice = self.validate_epoch()

            # Update learning rate
            self.scheduler.step(val_dice)

            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            epoch_time = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, "
                f"Time: {epoch_time:.2f}s\n"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                # Save best model
                torch.save(self.model.state_dict(), "best_model.pth")

        self.model.load_state_dict(torch.load("best_model.pth"))

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}
