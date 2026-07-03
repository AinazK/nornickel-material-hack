import os
import shutil
from pathlib import Path


def reorganize_dataset(root_dir: str = "data/train"):
    """Приводит структуру датасета к стандарту images/masks."""
    root = Path(root_dir)

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        # Проверяем, есть ли уже правильная структура
        if (class_dir / "images").exists() or (class_dir / "masks").exists():
            continue

        # Создаем новые директории
        img_dir = class_dir / "images"
        msk_dir = class_dir / "masks"
        img_dir.mkdir(exist_ok=True)
        msk_dir.mkdir(exist_ok=True)

        # Обрабатываем специальные случаи (silicate_ore с image/mask)
        for subdir_name in ["image", "mask"]:
            subdir = class_dir / subdir_name
            if subdir.exists():
                target = img_dir if subdir_name == "image" else msk_dir
                for f in subdir.iterdir():
                    if f.is_file():
                        shutil.move(str(f), str(target / f.name))
                subdir.rmdir()
                continue

        # Перемещаем файлы из корня класса (если есть)
        for f in class_dir.iterdir():
            if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tiff'}:
                # Эвристика: если в имени есть "mask" или "gt"
                if any(x in f.name.lower() for x in ['mask', 'gt', 'label']):
                    shutil.move(str(f), str(msk_dir / f.name))
                else:
                    shutil.move(str(f), str(img_dir / f.name))

    print("[Reorganize] Структура датасета приведена к стандарту")


if __name__ == "__main__":
    reorganize_dataset()