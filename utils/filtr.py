import cv2
import numpy as np
import os
from pathlib import Path


def normalize_image(image_path, output_path, target_brightness=127, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Приводит изображение к чб, выравнивает контраст с помощью CLAHE
    и корректирует среднюю яркость до целевого уровня.
    """
    # 1. Загружаем изображение в черно-белом формате (grayscale)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Ошибка загрузки: {image_path}")
        return False

    # 2. Применяем CLAHE для выравнивания локального контраста
    # Это подтягивает скрытые детали и выравнивает контрастность по всей площади
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl_img = clahe.apply(img)

    # 3. Корректируем общую яркость до целевого значения (target_brightness)
    # Вычисляем текущую среднюю яркость
    current_brightness = np.mean(cl_img)

    # Считаем коэффициент смещения
    brightness_diff = target_brightness - current_brightness

    # Применяем смещение яркости с защитой от выхода за границы [0, 255]
    normalized_img = cv2.convertScaleAbs(cl_img, alpha=1.0, beta=brightness_diff)

    # 4. Сохраняем результат
    cv2.imwrite(str(output_path), normalized_img)
    return True


def process_folder(input_dir, output_dir, target_brightness=127):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Поддерживаемые расширения
    extensions = ('*.JPG', '*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    image_files = []
    for ext in extensions:
        image_files.extend(input_path.glob(ext))

    if not image_files:
        print(f"В папки '{input_dir}' не найдено подходящих изображений.")
        return

    print(f"Найдено изображений для обработки: {len(image_files)}")

    success_count = 0
    for img_path in image_files:
        out_img_path = output_path / img_path.name
        if normalize_image(img_path, out_img_path, target_brightness):
            success_count += 1

    print(f"Обработка завершена! Успешно обработано: {success_count}/{len(image_files)}")
    print(f"Результаты сохранены в: {output_dir}")


# --- Точка входа ---
if __name__ == "__main__":
    # Укажи свои папки (можно использовать относительные или абсолютные пути)
    INPUT_FOLDER = "input_images"  # Папка с исходными фото (положи их сюда)
    OUTPUT_FOLDER = "ready_images"  # Сюда сохранятся чб результаты

    # target_brightness: 0 - абсолютно черный, 255 - белый. 127 — идеальная середина.
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, target_brightness=120)