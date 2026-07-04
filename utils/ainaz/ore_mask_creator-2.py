import os
import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


class OreMaskCreator:
    def __init__(self):
        self.image = None
        self.gray = None
        self.rgb = None

        self.masks = {
            "talc": None,
            "fine": None,
            "normal": None
        }

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Не удалось открыть {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        print(f"Размер: {self.image.shape}")
        print(f"Яркость: {self.gray.min()} - {self.gray.max()}")

    # ------------------------
    # Тёмные области (тальк)
    # ------------------------
    def create_talc_mask(self, threshold=50):

        _, mask = cv2.threshold(
            self.gray,
            threshold,
            255,
            cv2.THRESH_BINARY_INV
        )

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.masks["talc"] = mask

    # ------------------------
    # Тонкие срастания
    # ------------------------
    def create_fine_mask(self, threshold=85, min_area=500):
        _, binary = cv2.threshold(
            self.gray,
            threshold,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((5, 5), np.uint8)

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

        mask = np.zeros_like(binary)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255

        self.masks["fine"] = mask

    # ------------------------
    # Крупные светлые области
    # ------------------------
    def create_normal_mask(self):

        mask = cv2.inRange(
            self.gray,
            23,
            85
        )

        kernel = np.ones((3, 3), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.masks["normal"] = mask

    # ------------------------
    # Классификация руды
    # ------------------------
    def classify_ore(self, percentages):
        talc_percent = percentages.get("talc", 0.0)

        if talc_percent > 10:
            return "оталькованная руда (talk)"
        else:
            # Сравниваем доли обычных и тонких срастаний
            normal_percent = percentages.get("normal", 0.0)
            fine_percent = percentages.get("fine", 0.0)

            if normal_percent >= fine_percent:
                return "рядовая руда (normal)"
            else:
                return "труднообогатимая руда (fine)"

    # ------------------------
    # Отобразить
    # ------------------------
    def show(self):
        fig, ax = plt.subplots(2, 2, figsize=(12, 12))

        ax[0, 0].imshow(self.rgb)
        ax[0, 0].set_title("Original")

        colors = {
            "talc": [0, 0, 255],
            "fine": [255, 0, 0],
            "normal": [0, 255, 0]
        }

        positions = {
            "talc": (0, 1),
            "fine": (1, 0),
            "normal": (1, 1)
        }

        for name, mask in self.masks.items():
            row, col = positions[name]

            overlay = self.rgb.copy()
            if mask is not None:
                overlay[mask > 0] = colors[name]

            ax[row, col].imshow(overlay)
            ax[row, col].set_title(name)

        for a in ax.ravel():
            a.axis("off")

        plt.tight_layout()
        plt.show()

    # ------------------------
    # Сохранить
    # ------------------------
    def save(self, folder):
        Path(folder).mkdir(exist_ok=True, parents=True)

        for name, mask in self.masks.items():
            if mask is not None:
                cv2.imwrite(
                    os.path.join(folder, f"{name}.png"),
                    mask
                )


if __name__ == "__main__":
    # Папка, куда нужно положить исходные изображения
    INPUT_FOLDER = "ready_images"
    # Папка, где будут создаваться папки 1, 2, 3 и файл JSON
    OUTPUT_FOLDER = "output_results"

    Path(INPUT_FOLDER).mkdir(exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(exist_ok=True)

    # Надежный способ получения списка файлов без дубликатов
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    
    # Перебираем файлы в папке и фильтруем по расширению (приводя его к нижнему регистру)
    if os.path.exists(INPUT_FOLDER):
        for file in os.listdir(INPUT_FOLDER):
            if file.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(INPUT_FOLDER, file))

    if not image_paths:
        print(f"Положите изображения в папку '{INPUT_FOLDER}' и перезапустите скрипт.")
        exit()

    results_data = []

    # Обработка изображений из папки
    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        print(f"\n--- Обработка: {filename} (Папка {idx}) ---")
        
        creator = OreMaskCreator()

        try:
            creator.load_image(img_path)
        except ValueError as e:
            print(e)
            continue

        # Создаем маски
        creator.create_talc_mask()
        creator.create_fine_mask()
        creator.create_normal_mask()

        # Вычисление процентов
        percentages = {}
        total_pixels = creator.image.shape[0] * creator.image.shape[1]
        for name, mask in creator.masks.items():
            if mask is not None:
                count = np.count_nonzero(mask)
                percent = (count / total_pixels) * 100
                percentages[name] = round(percent, 2)

        # Классификация
        classification = creator.classify_ore(percentages)
        print(f"Классификация: {classification}")

        # Создаем индивидуальную папку (1, 2, 3...)
        current_out_folder = os.path.join(OUTPUT_FOLDER, str(idx))
        Path(current_out_folder).mkdir(exist_ok=True, parents=True)

        # Сохраняем черно-белые маски (используя оригинальный метод)
        creator.save(current_out_folder)

        # Сохраняем исходное изображение в эту папку
        original_img_path = os.path.join(current_out_folder, f"original_{filename}")
        cv2.imwrite(original_img_path, creator.image)

        # Сохраняем отдельные изображения с наложенными цветными масками
        # Используем BGR формат (Blue, Green, Red) для корректного сохранения через cv2
        mask_colors_bgr = {
            "talc": [255, 0, 0],   # Синий (BGR)
            "fine": [0, 0, 255],   # Красный (BGR)
            "normal": [0, 255, 0]  # Зеленый (BGR)
        }

        for name, mask in creator.masks.items():
            if mask is not None:
                # Создаем копию чистого исходного изображения для каждой маски
                overlay = creator.image.copy()
                
                # Накладываем только текущую маску
                overlay[mask > 0] = mask_colors_bgr[name]
                
                # Сохраняем отдельное изображение
                colored_mask_path = os.path.join(current_out_folder, f"{name}_colored_mask.jpg")
                cv2.imwrite(colored_mask_path, overlay)

        # Собираем данные для JSON
        results_data.append({
            "image_name": filename,
            "percentages": percentages,
            "classification": classification
        })

    # Сохранение итогового JSON файла
    json_path = os.path.join(OUTPUT_FOLDER, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Обработка завершена! Результаты сохранены в папку '{OUTPUT_FOLDER}', а сводка в {json_path}")