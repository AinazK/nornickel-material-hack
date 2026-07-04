import os
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
from typing import List, Tuple


class MassMaskCreator:
    """Массовое создание масок для всех изображений в датасете."""

    def __init__(self, talc_threshold: int = 23):
        self.talc_threshold = talc_threshold
        self.valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

    def create_talc_mask(self, image_path: str, output_path: str) -> bool:
        """
        Создать маску талька для одного изображения.
        Возвращает True если успешно.
        """
        try:
            # Загрузка изображения
            img = np.array(Image.open(image_path).convert("RGB"))
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            # Создание маски талька (тёмные области)
            _, mask = cv2.threshold(
                gray,
                self.talc_threshold,
                255,
                cv2.THRESH_BINARY_INV
            )

            # Морфологические операции для очистки шума
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Сохранение маски
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            Image.fromarray(mask).save(output_path)

            # Вычисление статистики
            total_pixels = mask.shape[0] * mask.shape[1]
            talc_pixels = np.count_nonzero(mask)
            percent = (talc_pixels / total_pixels) * 100

            return True, percent

        except Exception as e:
            print(f"  ❌ Ошибка при создании маски для {image_path}: {e}")
            return False, 0.0

    def process_dataset(self,
                        data_root: str,
                        masks_root: str,
                        class_names: List[str] = None) -> dict:
        """
        Обработать весь датасет и создать маски.

        Args:
            data_root: Корневая директория с данными (data/train)
            masks_root: Директория для сохранения масок (artifacts/masks)
            class_names: Список имён классов (папок) для обработки

        Returns:
            dict со статистикой обработки
        """
        data_root = Path(data_root)
        masks_root = Path(masks_root)

        stats = {
            "total_images": 0,
            "processed": 0,
            "failed": 0,
            "classes": {}
        }

        # Если классы не указаны, сканируем директорию
        if class_names is None:
            class_dirs = [d for d in data_root.iterdir() if d.is_dir()]
            class_names = [d.name for d in class_dirs]

        print(f"\n{'=' * 60}")
        print(f"📋 МАССОВОЕ СОЗДАНИЕ МАСОК")
        print(f"{'=' * 60}")
        print(f"📁 Источник: {data_root}")
        print(f"💾 Назначение: {masks_root}")
        print(f"📊 Классы: {', '.join(class_names)}")
        print(f"{'=' * 60}\n")

        for class_name in class_names:
            class_dir = data_root / class_name
            if not class_dir.exists():
                print(f"⚠️  Пропущено: {class_name} (директория не найдена)")
                continue

            # Поиск всех изображений в директории класса
            image_files = []
            for ext in self.valid_extensions:
                image_files.extend(class_dir.glob(f"*{ext}"))
                image_files.extend(class_dir.glob(f"*{ext.upper()}"))

            if not image_files:
                print(f"⚠️  Пропущено: {class_name} (нет изображений)")
                continue

            print(f"\n📂 Обработка класса: {class_name}")
            print(f"   Найдено изображений: {len(image_files)}")

            class_stats = {
                "total": len(image_files),
                "processed": 0,
                "failed": 0,
                "avg_talc_percent": 0.0
            }

            talc_percents = []

            # Создание масок для каждого изображения
            for idx, img_path in enumerate(image_files, 1):
                mask_path = masks_root / class_name / f"{img_path.stem}.png"

                success, talc_percent = self.create_talc_mask(str(img_path), str(mask_path))

                if success:
                    class_stats["processed"] += 1
                    talc_percents.append(talc_percent)
                    if idx % 10 == 0 or idx == len(image_files):
                        print(f"   ✓ Обработано: {idx}/{len(image_files)}")
                else:
                    class_stats["failed"] += 1

            # Вычисление среднего процента талька
            if talc_percents:
                class_stats["avg_talc_percent"] = sum(talc_percents) / len(talc_percents)

            stats["classes"][class_name] = class_stats
            stats["total_images"] += len(image_files)
            stats["processed"] += class_stats["processed"]
            stats["failed"] += class_stats["failed"]

            print(f"\n   ✅ Успешно: {class_stats['processed']}")
            print(f"   ❌ Ошибок: {class_stats['failed']}")
            print(f"   📊 Средний % талька: {class_stats['avg_talc_percent']:.2f}%")

        # Общая статистика
        print(f"\n{'=' * 60}")
        print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'=' * 60}")
        print(f"Всего изображений: {stats['total_images']}")
        print(f"✅ Успешно обработано: {stats['processed']}")
        print(f"❌ Ошибок: {stats['failed']}")
        print(f"📁 Маски сохранены в: {masks_root}")
        print(f"{'=' * 60}\n")

        return stats