import os
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
    def create_talc_mask(self, threshold=23):

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

        Path(folder).mkdir(exist_ok=True)

        for name, mask in self.masks.items():

            if mask is not None:
                cv2.imwrite(
                    os.path.join(folder, f"{name}.png"),
                    mask
                )


if __name__ == "__main__":

    creator = OreMaskCreator()

    creator.load_image("utils/ainaz/path_pred.JPG")

    creator.create_talc_mask()
    creator.create_fine_mask()
    creator.create_normal_mask()

    creator.show()

    creator.save("masks")