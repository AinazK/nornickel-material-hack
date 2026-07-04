import cv2
import numpy as np
import matplotlib.pyplot as plt


class OreMaskCreator:
    def __init__(self):
        self.image = None
        self.gray = None
        self.rgb = None
        self.mask = None

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Не удалось открыть {image_path}")

        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def create_mask(self, threshold=100, bright_expand_percent=5):
        """
        bright_expand_percent — насколько расширяем светлую область вниз по яркости
        (в процентах от общего диапазона серого)
        """

        min_g = int(self.gray.min())
        max_g = int(self.gray.max())
        range_g = max_g - min_g

        # расширение порога в сторону более тёмных значений
        offset = int(range_g * (bright_expand_percent / 100.0))

        adjusted_threshold = threshold + offset
        adjusted_threshold = min(255, adjusted_threshold)

        print(f"\nБазовый threshold: {threshold}")
        print(f"Расширение (%): {bright_expand_percent}")
        print(f"Сдвиг порога: +{offset}")
        print(f"Итоговый threshold: {adjusted_threshold}")

        # бинаризация
        _, self.mask = cv2.threshold(
            self.gray,
            adjusted_threshold,
            255,
            cv2.THRESH_BINARY
        )

        # Черная область
        black_pixels = self.gray[self.mask == 0]

        if len(black_pixels) > 0:
            print("\nЧерная область:")
            print(f"  Мин: {black_pixels.min()}")
            print(f"  Макс: {black_pixels.max()}")
            print(f"  Среднее: {black_pixels.mean():.2f}")

        # Белая область
        white_pixels = self.gray[self.mask == 255]

        if len(white_pixels) > 0:
            print("\nБелая область:")
            print(f"  Мин: {white_pixels.min()}")
            print(f"  Макс: {white_pixels.max()}")
            print(f"  Среднее: {white_pixels.mean():.2f}")

    def show(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        ax[0].imshow(self.rgb)
        ax[0].set_title("Оригинал")
        ax[0].axis("off")

        ax[1].imshow(self.mask, cmap="gray")
        ax[1].set_title("Бинарная маска")
        ax[1].axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    creator = OreMaskCreator()

    creator.load_image("utils/ainaz/path_pred.JPG")

    creator.create_mask(
        threshold=100,
        bright_expand_percent=8   # <- управляемое расширение
    )

    creator.show()