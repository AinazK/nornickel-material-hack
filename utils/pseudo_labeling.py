import os
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torchvision.models as tv_models
import torchvision.transforms as T

# Определяем корень проекта (на два уровня выше этого файла)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_features(model: torch.nn.Module, image: np.ndarray) -> np.ndarray:
    """Извлечение признаков из замороженного ResNet50."""
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze().numpy()


def generate_pseudo_masks(
        data_root: str = None,  # По умолчанию будет artifacts/preprocessed
        n_clusters: int = 4,
        output_dir: str = None  # По умолчанию будет artifacts/pseudo_masks
) -> None:
    """
    Автоматическая генерация псевдомасок через кластеризацию признаков.
    n_clusters: количество предполагаемых классов руд + фон.
    """
    # Используем абсолютные пути относительно корня проекта
    if data_root is None:
        data_root = PROJECT_ROOT / "artifacts" / "preprocessed"
    else:
        data_root = Path(data_root)

    if output_dir is None:
        output_dir = PROJECT_ROOT / "artifacts" / "pseudo_masks"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Замороженный энкодер для извлечения признаков
    print("[INFO] Загрузка ResNet50...")
    backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()
    print("[INFO] ResNet50 загружен")

    valid_ext = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

    # Проверка существования директории
    if not data_root.exists():
        print(f"[ERROR] Директория не найдена: {data_root}")
        print("[INFO] Доступные директории в artifacts/:")
        artifacts_dir = PROJECT_ROOT / "artifacts"
        if artifacts_dir.exists():
            for item in artifacts_dir.iterdir():
                if item.is_dir():
                    print(f"  - {item.name}")
        return

    class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    print(f"[PseudoLabel] Найдено {len(class_dirs)} классов в {data_root}")

    for class_dir in sorted(class_dirs):
        print(f"\n[INFO] Обработка класса: {class_dir.name}")

        class_output = output_dir / class_dir.name
        class_output.mkdir(parents=True, exist_ok=True)

        # Ищем изображения в поддиректории images/ или прямо в class_dir
        img_dir = class_dir / "images" if (class_dir / "images").exists() else class_dir
        images = [f for f in img_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_ext]

        print(f"  Найдено изображений: {len(images)}")

        if not images:
            print(f"  [WARN] Нет изображений в {class_dir}")
            continue

        feature_matrix = []
        img_paths = []
        img_sizes = []  # Сохраняем размеры для создания масок правильного размера

        # Сбор признаков
        for idx, img_path in enumerate(images):
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                img_sizes.append(img.shape[:2])  # (H, W)
                feat = extract_features(backbone, img)
                feature_matrix.append(feat)
                img_paths.append(img_path)

                if (idx + 1) % 10 == 0:
                    print(f"  Обработано: {idx + 1}/{len(images)}")

            except Exception as e:
                print(f"  [WARN] Ошибка {img_path.name}: {e}")

        if not feature_matrix:
            print(f"  [WARN] Не удалось извлечь признаки")
            continue

        print(f"  Извлечение признаков завершено: {len(feature_matrix)} изображений")
        feature_matrix = np.vstack(feature_matrix)
        print(f"  Размерность признаков: {feature_matrix.shape}")

        # Нормализация и кластеризация
        scaler = StandardScaler()
        feature_scaled = scaler.fit_transform(feature_matrix)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feature_scaled)

        print(f"  Кластеризация завершена. Распределение меток:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"    Класс {label}: {count} изображений")

        # Сохранение псевдомасок с правильными размерами
        masks_created = 0
        for img_path, label, size in zip(img_paths, labels, img_sizes):
            try:
                # Создаём маску того же размера, что и исходное изображение
                h, w = size
                mask = np.full((h, w), label, dtype=np.uint8)
                mask_path = class_output / f"{img_path.stem}.png"
                Image.fromarray(mask).save(mask_path)
                masks_created += 1
            except Exception as e:
                print(f"  [ERROR] Не удалось сохранить маску для {img_path.name}: {e}")

        print(f"  ✓ Создано масок: {masks_created}")

    print(f"\n{'=' * 60}")
    print(f"[PseudoLabel] Псевдомаски сохранены в: {output_dir}")
    print(f"[INFO] Для обучения используйте: python main.py")
    print(f"{'=' * 60}")

    # Проверка результата
    if output_dir.exists():
        total_masks = sum(1 for _ in output_dir.rglob("*.png"))
        print(f"[INFO] Всего масок создано: {total_masks}")


if __name__ == "__main__":
    generate_pseudo_masks()