import cv2
import numpy as np

class OreProcessor:
    def __init__(self):
        self.image = None
        self.gray = None
        self.rgb = None
        self.masks = {"talc": None, "fine": None, "normal": None}

    def load_from_bytes(self, file_bytes):
        """Загрузка изображения из байтов (Streamlit uploader)"""
        # Декодируем байты в формат OpenCV
        encoded_img = np.frombuffer(file_bytes, dtype=np.uint8)
        self.image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
        
        if self.image is None:
            raise ValueError("Не удалось декодировать изображение.")
            
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def apply_clahe_preprocessing(self, target_brightness=120, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Предобработка методом CLAHE и выравнивание яркости (из твоего 2-го файла)"""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl_img = clahe.apply(self.gray)
        
        current_brightness = np.mean(cl_img)
        brightness_diff = target_brightness - current_brightness
        
        # Обновляем grayscale сбалансированной версией
        self.gray = cv2.convertScaleAbs(cl_img, alpha=1.0, beta=brightness_diff)

    def create_talc_mask(self, threshold=50):
        _, mask = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.masks["talc"] = mask

    def create_fine_mask(self, threshold=85, min_area=500):
        _, binary = cv2.threshold(self.gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        mask = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                mask[labels == i] = 255
        self.masks["fine"] = mask

    def create_normal_mask(self):
        mask = cv2.inRange(self.gray, 23, 85)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.masks["normal"] = mask

    def get_statistics(self):
        """Расчет процентов содержания элементов"""
        percentages = {}
        total_pixels = self.image.shape[0] * self.image.shape[1]
        
        for name, mask in self.masks.items():
            if mask is not None:
                count = np.count_nonzero(mask)
                percent = (count / total_pixels) * 100
                percentages[name] = round(percent, 2)
        
        # Считаем остаток породы (все, что не попало в маски)
        combined_masks = np.zeros_like(self.gray)
        for mask in self.masks.values():
            if mask is not None:
                combined_masks = cv2.bitwise_or(combined_masks, mask)
        
        waste_pixels = total_pixels - np.count_nonzero(combined_masks)
        percentages["waste"] = round((waste_pixels / total_pixels) * 100, 2)
        
        return percentages

    def classify_ore(self, percentages):
        talc_percent = percentages.get("talc", 0.0)
        if talc_percent > 10:
            return "Оталькованная руда (Talc)"
        
        normal_percent = percentages.get("normal", 0.0)
        fine_percent = percentages.get("fine", 0.0)
        if normal_percent >= fine_percent:
            return "Рядовая руда (Normal)"
        else:
            return "Труднообогатимая руда (Fine)"

    def get_combined_overlay(self):
        """Создает финальное изображение, где все маски наложены разными цветами"""
        overlay = self.rgb.copy()
        
        # Цвета в формате RGB
        colors = {
            "talc": [0, 0, 255],     # Синий
            "fine": [255, 0, 0],     # Красный
            "normal": [0, 255, 0]    # Зеленый
        }
        
        for name, mask in self.masks.items():
            if mask is not None:
                overlay[mask > 0] = colors[name]
                
        return overlay