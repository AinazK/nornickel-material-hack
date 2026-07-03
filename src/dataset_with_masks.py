import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, List, Tuple
from pathlib import Path


class OreSegmentationDatasetWithMasks(Dataset):
    """
    Датасет для обучения с масками.
    Ожидает структуру: class_name/images/ и class_name/masks/
    """

    def __init__(
            self,
            data_root: str,
            transforms: Optional[Callable] = None,
            class_mapping: Optional[dict] = None
    ):
        self.transforms = transforms
        self.valid_ext = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

        self.class_mapping = class_mapping or {
            "refractory_ore": 1,
            "regular_ore": 2,
            "silicate_ore": 3,
            "thin_ore": 4
        }

        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []
        self.labels: List[int] = []

        self._scan_dataset(data_root)

    def _scan_dataset(self, root_dir: str):
        """Рекурсивный обход с поиском пар image/mask."""
        root = Path(root_dir)

        for class_dir in root.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_id = self.class_mapping.get(class_name, 0)

            # Ищем директории images и masks
            img_dir = class_dir / "images"
            msk_dir = class_dir / "masks"

            if not img_dir.exists() or not msk_dir.exists():
                print(f"[WARN] Пропущена директория {class_dir}: нет images/ или masks/")
                continue

            # Сопоставление файлов по имени
            img_files = {f.stem: f for f in img_dir.iterdir()
                         if f.suffix.lower() in self.valid_ext}
            msk_files = {f.stem: f for f in msk_dir.iterdir()
                         if f.suffix.lower() in self.valid_ext}

            for stem, img_path in img_files.items():
                if stem in msk_files:
                    self.image_paths.append(img_path)
                    self.mask_paths.append(msk_files[stem])
                    self.labels.append(class_id)
                else:
                    print(f"[WARN] Нет маски для {img_path.name}")

        print(f"[Dataset] Загружено {len(self.image_paths)} пар изображений с масками")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(self.image_paths[idx]).convert("RGB"), dtype=np.uint8)
        msk = np.array(Image.open(self.mask_paths[idx]).convert("L"), dtype=np.uint8)

        # Нормализация маски
        msk = np.clip(msk // 255, 0, 1).astype(np.uint8) * self.labels[idx]

        if self.transforms:
            augmented = self.transforms(image=img, mask=msk)
            img, msk = augmented["image"], augmented["mask"]
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            msk = torch.from_numpy(msk).long()

        return img, msk