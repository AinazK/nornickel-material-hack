import os
import numpy as np
from PIL import Image
from typing import Tuple


def create_patches(
        src_img_dir: str,
        src_mask_dir: str,
        dst_img_dir: str,
        dst_mask_dir: str,
        patch_size: Tuple[int, int] = (1024, 1024),
        overlap: int = 128
) -> None:
    """Разбиение панорамных изображений и масок на патчи с заданным перекрытием."""
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_mask_dir, exist_ok=True)

    img_files = [f for f in os.listdir(src_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))]

    step_x = patch_size[0] - overlap
    step_y = patch_size[1] - overlap

    for img_name in img_files:
        img_path = os.path.join(src_img_dir, img_name)
        mask_path = os.path.join(src_mask_dir, img_name)

        if not os.path.exists(mask_path):
            continue  # Пропуск, если маска отсутствует (нарушение консистентности)

        with Image.open(img_path) as img:
            img_arr = np.array(img.convert("RGB"), dtype=np.uint8)
        with Image.open(mask_path) as msk:
            msk_arr = np.array(msk.convert("L"), dtype=np.uint8)

        h, w, _ = img_arr.shape
        base_name = os.path.splitext(img_name)[0]

        patch_idx = 0
        for y in range(0, h, step_y):
            for x in range(0, w, step_x):
                y1, y2 = y, min(y + patch_size[1], h)
                x1, x2 = x, min(x + patch_size[0], w)

                # Пропуск патчей, не достигающих минимального размера
                if (y2 - y1) < patch_size[1] // 2 or (x2 - x1) < patch_size[0] // 2:
                    continue

                patch_img = img_arr[y1:y2, x1:x2]
                patch_msk = msk_arr[y1:y2, x1:x2]

                patch_name = f"{base_name}_x{x}_y{y}.jpg"
                Image.fromarray(patch_img).save(os.path.join(dst_img_dir, patch_name))

                # Маски сохраняем в PNG для сохранения точных классов
                Image.fromarray(patch_msk).save(os.path.join(dst_mask_dir, patch_name.replace('.jpg', '.png')))
                patch_idx += 1

    print(f"[Patching] Обработано {len(img_files)} панорам. Сгенерировано патчей: {patch_idx * len(img_files)}")