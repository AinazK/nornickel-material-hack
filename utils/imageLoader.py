from PIL import Image
import numpy as np
import io
from utils.imageData import ImageData


def load_image(uploaded_file):
    """ Преобразует файл из Streamlit в ImageData """

    # 1. читаем байты
    file_bytes = uploaded_file.read()

    # 2. размер файла
    file_size = len(file_bytes)

    # 3. открываем изображение
    image = Image.open(io.BytesIO(file_bytes))

    # 4. приводим к RGB (чтобы не было проблем)
    image = image.convert("RGB")

    # 5. переводим в numpy
    image_np = np.array(image)

    # 6. размеры
    height, width = image_np.shape[:2]

    # 7. формат
    image_format = image.format if image.format else uploaded_file.type

    # 8. создаём объект
    return ImageData(
        image=image_np,
        filename=uploaded_file.name,
        width=width,
        height=height,
        image_format=image_format,
        file_size=file_size
    )