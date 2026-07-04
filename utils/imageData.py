from dataclasses import dataclass
import numpy as np

@dataclass
class ImageData:
    image: np.ndarray   # изображение в виде массива чисел
    filename: str       # имя файла
    width: int          # ширина в пикселях
    height: int         # высота в пикселях
    image_format: str   # формат файла (JPEG / PNG / TIFF)
    file_size: int      # размер в байтах