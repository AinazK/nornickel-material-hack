import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import os
from pathlib import Path


class OreMaskCreator:
    def __init__(self, image_path=None):
        """
        Инициализация инструмента для создания масок рудных текстур
        
        Параметры:
        - image_path: путь к изображению (опционально)
        """
        self.image = None
        self.image_path = image_path
        self.masks = {
            'talc': None,              # Синяя маска - тёмные/чёрные области (тальк)
            'fine_intergrowth': None,  # Красная маска - тонкие срастания
            'normal_intergrowth': None # Зеленая маска - обычные срастания
        }
        self.current_mask = None
        self.drawing = False
        self.last_point = None
        self.brush_size = 20
        self.threshold_value = 127
        
        if image_path and os.path.exists(image_path):
            self.load_image(image_path)
    
    def load_image(self, path):
        """Загрузка изображения"""
        self.image = cv2.imread(path)
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {path}")
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.image_path = path
        print(f"Изображение загружено: {path}")
        print(f"Размер: {self.image.shape}")
        print(f"Диапазон яркости: min={self.image_gray.min()}, max={self.image_gray.max()}")
    
    def create_talc_mask(self, method='threshold', **kwargs):
        """
        Создание синей маски для талька (тёмные/чёрные области)
        
        Методы:
        - 'threshold': пороговая обработка (выделение тёмных пикселей)
        - 'adaptive': адаптивная пороговая обработка
        - 'manual': ручная разметка
        - 'color_based': выделение по цвету (HSV)
        """
        if self.image is None:
            raise ValueError("Сначала загрузите изображение")
        
        if method == 'threshold':
            # Для выделения ТЁМНЫХ областей используем обычный THRESH_BINARY
            # (пиксели НИЖЕ порога становятся белыми на маске)
            thresh = kwargs.get('threshold', 80)  # Низкий порог для чёрных оттенков
            
            # Инвертируем: тёмные пиксели (0-thresh) -> 255, светлые -> 0
            _, mask = cv2.threshold(self.image_gray, thresh, 255, cv2.THRESH_BINARY_INV)
            
            # Морфологические операции для удаления шума
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
        elif method == 'adaptive':
            # Адаптивный порог для неравномерного освещения
            block_size = kwargs.get('block_size', 15)
            C = kwargs.get('C', 5)
            mask = cv2.adaptiveThreshold(
                self.image_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,  # Инверсия для тёмных областей
                block_size, C
            )
            
        elif method == 'color_based':
            # Выделение по цвету в HSV пространстве
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            
            # Тёмные области: низкая яркость (V)
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, kwargs.get('max_brightness', 60)])
            
            mask = cv2.inRange(hsv, lower_dark, upper_dark)
            
            # Морфологическая очистка
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
        elif method == 'manual':
            print("Ручная разметка: используйте метод draw_mask()")
            return None
        
        self.masks['talc'] = mask
        return mask
    
    def create_fine_intergrowth_mask(self, method='edge_based', **kwargs):
        """
        Создание красной маски для тонких срастаний
        (сульфиды, замещённые нерудной фазой)
        """
        if self.image is None:
            raise ValueError("Сначала загрузите изображение")
        
        if method == 'edge_based':
            # Обнаружение краев для выявления тонких структур
            blurred = cv2.GaussianBlur(self.image_gray, (5, 5), 0)
            edges = cv2.Canny(blurred,
                            kwargs.get('low_threshold', 50),
                            kwargs.get('high_threshold', 150))
            
            # Дилатация для утолщения тонких линий
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(edges, kernel, iterations=kwargs.get('iterations', 2))
            
        elif method == 'texture':
            # Анализ текстуры для выявления замещенных областей
            img_float = self.image_gray.astype(np.float32)
            mean = cv2.boxFilter(img_float, -1, (15, 15))
            sqr_mean = cv2.boxFilter(img_float**2, -1, (15, 15))
            variance = sqr_mean - mean**2
            std_dev = np.sqrt(np.abs(variance))
            
            std_dev = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX)
            std_dev = std_dev.astype(np.uint8)
            
            _, mask = cv2.threshold(std_dev,
                                   kwargs.get('threshold', 100),
                                   255, cv2.THRESH_BINARY)
        
        self.masks['fine_intergrowth'] = mask
        return mask
    
    def create_normal_intergrowth_mask(self, method='blob_detection', **kwargs):
        """
        Создание зеленой маски для обычных срастаний
        (крупные изолированные сульфиды - светлые области)
        """
        if self.image is None:
            raise ValueError("Сначала загрузите изображение")
        
        if method == 'blob_detection':
            # Выделение СВЕТЛЫХ областей (сульфиды)
            _, thresh = cv2.threshold(self.image_gray,
                                     kwargs.get('threshold', 150),
                                     255, cv2.THRESH_BINARY)
            
            # Морфологические операции
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Удаление мелких объектов
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
            mask = np.zeros_like(thresh)
            min_area = kwargs.get('min_area', 500)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area > min_area:
                    mask[labels == i] = 255
            
        elif method == 'contour':
            blurred = cv2.GaussianBlur(self.image_gray, (5, 5), 0)
            _, thresh = cv2.threshold(blurred,
                                     kwargs.get('threshold', 150),
                                     255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(thresh)
            min_area = kwargs.get('min_area', 500)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    cv2.drawContours(mask, [contour], -1, 255, -1)
        
        self.masks['normal_intergrowth'] = mask
        return mask
    
    def draw_mask(self, mask_type, color=(255, 255, 255)):
        """
        Интерактивная ручная разметка маски
        
        Параметры:
        - mask_type: тип маски ('talc', 'fine_intergrowth', 'normal_intergrowth')
        - color: цвет кисти (по умолчанию белый)
        """
        if self.image is None:
            raise ValueError("Сначала загрузите изображение")
        
        if self.masks[mask_type] is None:
            self.masks[mask_type] = np.zeros(self.image_gray.shape, dtype=np.uint8)
        
        mask = self.masks[mask_type].copy()
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal mask, drawing, last_point
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                last_point = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.line(mask, last_point, (x, y), color, self.brush_size)
                    last_point = (x, y)
            
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
        
        drawing = False
        last_point = None
        
        display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if self.image is not None:
            overlay = self.image.copy()
            overlay[mask > 0] = color
            display_img = cv2.addWeighted(self.image, 0.7, overlay, 0.3, 0)
        
        cv2.namedWindow(f'Drawing {mask_type} mask')
        cv2.setMouseCallback(f'Drawing {mask_type} mask', mouse_callback)
        
        print(f"Рисование маски: {mask_type}")
        print("ЛКМ - рисовать, 'q' - выйти, 'c' - очистить, 's' - сохранить")
        
        while True:
            cv2.imshow(f'Drawing {mask_type} mask', display_img)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                mask = np.zeros(self.image_gray.shape, dtype=np.uint8)
                display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            elif key == ord('s'):
                self.masks[mask_type] = mask
                print(f"Маска {mask_type} сохранена")
                break
        
        cv2.destroyAllWindows()
        self.masks[mask_type] = mask
        return mask
    
    def visualize_masks(self, save_path=None):
        """Визуализация всех масок"""
        if self.image is None:
            raise ValueError("Сначала загрузите изображение")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        
        # Оригинальное изображение
        axes[0, 0].imshow(self.image_rgb)
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')
        
        # Маска талька (синяя) - тёмные области
        if self.masks['talc'] is not None:
            overlay = self.image_rgb.copy()
            # Синий цвет в RGB: [0, 0, 255]
            overlay[self.masks['talc'] > 0] = [0, 0, 255]
            axes[0, 1].imshow(overlay)
            axes[0, 1].set_title('Talc Mask - Dark Areas (Blue)', fontsize=14)
        axes[0, 1].axis('off')
        
        # Маска тонких срастаний (красная)
        if self.masks['fine_intergrowth'] is not None:
            overlay = self.image_rgb.copy()
            overlay[self.masks['fine_intergrowth'] > 0] = [255, 0, 0]
            axes[1, 0].imshow(overlay)
            axes[1, 0].set_title('Fine Intergrowth Mask (Red)', fontsize=14)
        axes[1, 0].axis('off')
        
        # Маска обычных срастаний (зеленая)
        if self.masks['normal_intergrowth'] is not None:
            overlay = self.image_rgb.copy()
            overlay[self.masks['normal_intergrowth'] > 0] = [0, 255, 0]
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('Normal Intergrowth Mask (Green)', fontsize=14)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Визуализация сохранена: {save_path}")
        
        plt.show()
    
    def save_masks(self, output_dir):
        """Сохранение всех масок"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for mask_type, mask in self.masks.items():
            if mask is not None:
                output_path = os.path.join(output_dir, f'{mask_type}_mask.png')
                cv2.imwrite(output_path, mask)
                print(f"Маска сохранена: {output_path}")
        
        # Сохранение цветной композитной маски
        if any(m is not None for m in self.masks.values()):
            composite = np.zeros((*self.image_gray.shape, 3), dtype=np.uint8)
            # BGR формат для OpenCV
            if self.masks['talc'] is not None:
                composite[self.masks['talc'] > 0] = [255, 0, 0]      # Синий в BGR
            if self.masks['fine_intergrowth'] is not None:
                composite[self.masks['fine_intergrowth'] > 0] = [0, 0, 255]  # Красный в BGR
            if self.masks['normal_intergrowth'] is not None:
                composite[self.masks['normal_intergrowth'] > 0] = [0, 255, 0]  # Зеленый в BGR
            
            output_path = os.path.join(output_dir, 'composite_mask.png')
            cv2.imwrite(output_path, composite)
            print(f"Композитная маска сохранена: {output_path}")


def batch_process(directory, output_dir, **kwargs):
    """
    Пакетная обработка изображений в директории
    
    Параметры:
    - directory: путь к директории с изображениями
    - output_dir: путь к директории для сохранения масок
    """
    creator = OreMaskCreator()
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_path = os.path.join(directory, filename)
            print(f"\nОбработка: {filename}")
            
            try:
                creator.load_image(image_path)
                
                # Создание масок с параметрами по умолчанию
                # Для тёмных областей используем низкий порог
                creator.create_talc_mask(method='threshold', threshold=80)
                creator.create_fine_intergrowth_mask(method='edge_based')
                creator.create_normal_intergrowth_mask(method='blob_detection', threshold=150)
                
                # Сохранение результатов
                file_output_dir = os.path.join(output_dir, Path(filename).stem)
                creator.save_masks(file_output_dir)
                creator.visualize_masks(save_path=os.path.join(file_output_dir, 'visualization.png'))
                
            except Exception as e:
                print(f"Ошибка обработки {filename}: {e}")


# Пример использования
if __name__ == "__main__":
    # Создание инструмента
    creator = OreMaskCreator()
    
    # Загрузка изображения
    image_path = "path_pred.JPG"  # Замените на путь к вашему изображению
    
    if os.path.exists(image_path):
        creator.load_image(image_path)
        
        # Автоматическое создание масок
        # ГЛАВНОЕ ИЗМЕНЕНИЕ: выделение тёмных/чёрных областей синей маской
        # Используем низкий порог (80) и THRESH_BINARY_INV
        creator.create_talc_mask(method='threshold', threshold=80)
        
        creator.create_fine_intergrowth_mask(method='edge_based',
                                            low_threshold=50,
                                            high_threshold=150)
        creator.create_normal_intergrowth_mask(method='blob_detection',
                                              threshold=150,
                                              min_area=500)
        
        # Визуализация
        creator.visualize_masks(save_path='masks_visualization.png')
        
        # Сохранение масок
        creator.save_masks('output_masks')
        
        print("\nМаски созданы и сохранены!")
        print("\nСиняя маска выделяет тёмные/чёрные области (тальк)")
        print("Красная маска - тонкие срастания")
        print("Зеленая маска - обычные срастания (светлые сульфиды)")
    else:
        print(f"Изображение не найдено: {image_path}")
        print("\nПример использования:")
        print("1. Замените 'path_pred.JPG' на путь к вашему изображению")
        print("2. Или используйте интерактивную разметку:")
        print("   creator.draw_mask('talc', color=(255, 0, 0))")