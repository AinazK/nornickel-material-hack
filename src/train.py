import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import os

from dataset import OreSegmentationDataset
from transforms import get_train_transforms, get_val_transforms

# Фиксация детерминированности
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 5  # Укажите точное количество классов (включая фон, если он не 0)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights="imagenet",
        in_channels=3,
        classes=NUM_CLASSES
    ).to(device)

    # Многоклассовые функции потерь
    criterion = smp.losses.DiceLoss(mode="multiclass")
    # Альтернатива: nn.CrossEntropyLoss(weight=class_weights) при дисбалансе
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)
    scaler = GradScaler()

    train_ds = OreSegmentationDataset("dataset", split="train", transforms=get_train_transforms())
    test_ds = OreSegmentationDataset("dataset", split="test", transforms=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    epochs = 25
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # Валидация на панорамных патчах
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)
                val_loss += criterion(model(images), masks).item()

        val_loss /= len(test_loader)
        scheduler.step()
        print(f"[Epoch {epoch + 1}] Train: {epoch_loss / len(train_loader):.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoints/best_unet_resnet50.pth")


if __name__ == "__main__":
    main()