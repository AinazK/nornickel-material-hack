import cv2
import numpy as np
from typing import Tuple

def preprocess_geological_image(
    img: np.ndarray,
    target_size: Tuple[int, int] = (1024, 1024),
    clahe_clip: float = 2.0,
    clahe_grid: int = 8,
    denoise_sigma: float = 50.0
) -> np.ndarray:
    """
    Детерминированная предобработка минералогических изображений.
    - Шумоподавление: двусторонний фильтр (сохраняет резкие границы фаз)
    - Нормализация освещения и контраста: CLAHE в канале L (LAB)
    - Масштабирование: сохранение пропорций с дополнением до target_size
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    # 1. Шумоподавление с сохранением краёв
    denoised = cv2.bilateralFilter(
        img, d=5, sigmaColor=denoise_sigma, sigmaSpace=denoise_sigma
    )

    # 2. Адаптивная коррекция контраста и освещения (CLAHE)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
    lab[:, :, 0] = clahe.apply(l_channel)
    corrected = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # 3. Масштабирование с сохранением аспекта и валидным заполнением
    h, w, _ = corrected.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(corrected, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_h = target_size[1] - new_h
    pad_w = target_size[0] - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    # Дополнение средним цветом изображения вместо нулей для исключения артефактов
    mean_color = [int(c) for c in np.mean(resized, axis=(0, 1))]
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=mean_color
    )

    return padded