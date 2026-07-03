import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import random

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import OreSegmentationDataset
from transforms import get_train_transforms, get_val_transforms


def main():
    CONFIG = {
        "data_root": str(PROJECT_ROOT / "artifacts" / "pseudo_masks"),
        "masks_dir": str(PROJECT_ROOT / "artifacts" / "pseudo_masks"),
        "batch_size": 1,
        "num_workers": 0,
        "num_epochs": 3,
        "num_classes": 4,
        "learning_rate": 1e-3,
        "device": "cpu",
        "checkpoint_dir": str(PROJECT_ROOT / "artifacts" / "checkpoints"),
        "image_size": 256,
        "max_samples": 200,  # Быстрый тест на подвыборке
    }

    print("=" * 60)
    print("🚀 БЫСТРЫЙ СТАРТ (CPU-оптимизировано)")
    print("=" * 60)
    print(f"📊 Размер изображений: {CONFIG['image_size']}x{CONFIG['image_size']}")
    print(f"📦 Batch size: {CONFIG['batch_size']}")
    print(f"🔄 Эпох: {CONFIG['num_epochs']}")
    print(f"📁 Макс. образцов: {CONFIG['max_samples']}")
    print("=" * 60)

    # Загрузка датасета
    full_dataset = OreSegmentationDataset(
        data_root=CONFIG["data_root"],
        masks_dir=CONFIG["masks_dir"],
        transforms=get_train_transforms(CONFIG["image_size"]),
        recursive=True
    )

    # Берём подвыборку для быстрого теста
    if len(full_dataset) > CONFIG["max_samples"]:
        indices = random.sample(range(len(full_dataset)), CONFIG["max_samples"])
        dataset = Subset(full_dataset, indices)
        print(f"✅ Используется подвыборка: {len(dataset)} из {len(full_dataset)}")
    else:
        dataset = full_dataset

    # Разделение train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"📈 Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"📉 Батчей за эпоху: {len(train_ds) // CONFIG['batch_size']}")
    print("=" * 60)

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"]
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"]
    )

    # ЛЁГКАЯ МОДЕЛЬ (MobileNet вместо ResNet50)
    print("⚙️ Загрузка модели (MobileNetV2 - лёгкая)...")
    model = smp.Unet(
        encoder_name="mobilenet_v2",  # В 5-10 раз быстрее ResNet50
        encoder_weights="imagenet",
        in_channels=3,
        classes=CONFIG["num_classes"]
    ).to(CONFIG["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    Path(CONFIG["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    print("\n🎯 НАЧАЛО ОБУЧЕНИЯ...\n")

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks, names) in enumerate(train_loader):
            images = images.to(CONFIG["device"])
            masks = masks.long().to(CONFIG["device"])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch + 1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images = images.to(CONFIG["device"])
                masks = masks.long().to(CONFIG["device"])
                val_loss += criterion(model(images), masks).item()

        avg_train = epoch_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)

        print(f"\n✅ Epoch {epoch + 1}/{CONFIG['num_epochs']} завершена")
        print(f"   Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # === СОХРАНЯЕМ ТОЛЬКО ЛУЧШУЮ МОДЕЛЬ ===
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1

            # Удаляем старые чекпоинты (опционально)
            for old_file in Path(CONFIG["checkpoint_dir"]).glob("model_epoch_*.pth"):
                old_file.unlink()

            # Сохраняем лучшую
            best_path = Path(CONFIG["checkpoint_dir"]) / "best_model.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": CONFIG
            }, best_path)
            print(f"   🏆 НОВАЯ ЛУЧШАЯ МОДЕЛЬ! Сохранено: best_model.pth")
        else:
            print(f"   ⏭️ Пропуск сохранения (val_loss {avg_val:.4f} > best {best_val_loss:.4f})")

        print("-" * 60)

    print(f"\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print(f"🏆 Лучшая эпоха: {best_epoch} (Val Loss: {best_val_loss:.4f})")
    print(f"📁 Модель: {CONFIG['checkpoint_dir']}/best_model.pth")
    print("=" * 60)


if __name__ == "__main__":
    import segmentation_models_pytorch as smp

    main()