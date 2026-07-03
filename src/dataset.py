import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, List, Tuple
from pathlib import Path


class OreSegmentationDataset(Dataset):
    """
    Универсальный датасет для сегментации руд.
    Поддерживает:
    1. Структуру обучения: class/images/ + class/masks/
    2. Плоскую структуру: class/*.png (для псевдомасок)
    3. Рекурсивный обход поддиректорий
    """

    def __init__(
            self,
            data_root: str,
            masks_dir: Optional[str] = None,
            transforms: Optional[Callable] = None,
            class_id: Optional[int] = None,  # Если None, используется имя папки как ID
            recursive: bool = True  # Рекурсивный поиск по подпапкам
    ):
        self.transforms = transforms
        self.class_id = class_id
        self.valid_ext = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

        self.data_root = Path(data_root)
        self.masks_root = Path(masks_dir) if masks_dir else None

        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []
        self.labels: List[int] = []

        self._scan_directory(recursive)

        # Гарантируем наличие __len__ через инициализацию списков
        print(f"[Dataset] Инициализировано: {len(self.image_paths)} образцов")

    def _scan_directory(self, recursive: bool = True):
        """Рекурсивное сканирование с поддержкой разных структур."""
        if not self.data_root.exists():
            print(f"[ERROR] Директория не найдена: {self.data_root}")
            return

        print(f"[Dataset] Сканирование: {self.data_root}")

        # Генератор файлов: rglob для рекурсии, glob для текущего уровня
        file_iterator = self.data_root.rglob("*") if recursive else self.data_root.glob("*")

        processed_classes = set()

        for file_path in file_iterator:
            if not file_path.is_file() or file_path.suffix.lower() not in self.valid_ext:
                continue

            # Определяем класс по имени родительской директории
            class_dir = file_path.parent.name
            if class_dir in {".", "images", "masks"}:
                class_dir = self.data_root.name

            # Логика поиска соответствующей маски
            mask_path = None

            # Сценарий А: Структура с images/masks
            if "images" in file_path.parts:
                # Извлекаем относительный путь от images/
                rel_path = file_path.relative_to(self.data_root / "images")
                potential_mask = self.masks_root / "masks" / rel_path if self.masks_root else None
                if potential_mask and potential_mask.exists():
                    mask_path = potential_mask

            # Сценарий Б: Плоская структура или псевдомаски
            elif self.masks_root and self.masks_root.exists():
                # Ищем маску с тем же именем в mirrors-структуре
                if recursive and self.masks_root != self.data_root:
                    # Пытаемся сохранить иерархию
                    rel_path = file_path.relative_to(self.data_root)
                    potential_mask = self.masks_root / rel_path
                else:
                    # Плоский поиск
                    potential_mask = self.masks_root / file_path.name

                if potential_mask.exists():
                    mask_path = potential_mask

            # Если маска найдена или это режим инференса (mask_path может быть None)
            self.image_paths.append(file_path)
            self.mask_paths.append(mask_path if mask_path else file_path)  # Для псевдомасок: image == mask

            # Назначение лейбла
            if self.class_id is not None:
                self.labels.append(self.class_id)
            else:
                # Хэш имени класса как ID (для псевдомасок)
                self.labels.append(hash(class_dir) % 255)

            if class_dir not in processed_classes:
                print(f"  Найдено: {class_dir}/")
                processed_classes.add(class_dir)

    def __len__(self) -> int:
        """Гарантированное возвращение целого числа."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path = self.image_paths[idx]
        msk_path = self.mask_paths[idx]

        # Загрузка изображения
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Загрузка маски
        if msk_path.exists():
            msk = np.array(Image.open(msk_path).convert("L"), dtype=np.uint8)
        else:
            msk = np.zeros(img.shape[:2], dtype=np.uint8)

        # Ресайз маски под изображение
        if msk.shape != img.shape[:2]:
            msk = np.array(Image.fromarray(msk).resize(
                (img.shape[1], img.shape[0]), Image.NEAREST
            ), dtype=np.uint8)

        # Применение аугментаций
        if self.transforms:
            augmented = self.transforms(image=img, mask=msk)
            img_tensor = augmented["image"]
            msk_tensor = augmented["mask"]
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            msk_tensor = torch.from_numpy(msk)  # Временный тензор

        # === КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: приведение маски к LongTensor ===
        # 1. Убедиться, что данные целочисленные
        if msk_tensor.dtype in [torch.float32, torch.float64]:
            msk_tensor = msk_tensor.round()

        # 2. Привести к torch.long (int64)
        msk_tensor = msk_tensor.long()

        # 3. Обрезать значения за пределами допустимого диапазона классов
        # (защита от артефактов аугментаций)
        msk_tensor = torch.clamp(msk_tensor, 0, self.num_classes - 1 if hasattr(self, 'num_classes') else 255)

        return img_tensor, msk_tensor, img_path.name