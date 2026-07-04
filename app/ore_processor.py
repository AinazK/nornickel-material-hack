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

    # ------------------------
    # Тёмные области (тальк)
    # ------------------------
    def create_talc_mask(
        self,
        outlier_std_margin=1.2,
        max_talc_percent=30,
        min_brightness=0,
        use_local_contrast=True,
        local_kernel_size=51,
        local_contrast_margin=30,
        merge_distance=30,
        max_hole_area=500,
        smooth_size=27,
        max_component_percent=2,
        homogeneity_std_threshold=10,
        min_area=0
    ):
        # -------------------------
        # 0. K-means: определяем рудную и нерудную область
        # -------------------------
        pixels = self.gray.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)

        _, labels, centers = cv2.kmeans(
            pixels, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()
        centers = centers.flatten()

        if centers[0] < centers[1]:
            non_ore_label, ore_label = 0, 1
        else:
            non_ore_label, ore_label = 1, 0

        non_ore_pixels = pixels[labels == non_ore_label]
        non_ore_mean = float(non_ore_pixels.mean())
        non_ore_std = float(non_ore_pixels.std())

        threshold = non_ore_mean - outlier_std_margin * non_ore_std

        # -------------------------
        # Потолок: тальк не должен занимать больше max_talc_percent
        # -------------------------
        percentile_cap = float(np.percentile(self.gray, max_talc_percent))
        if threshold > percentile_cap:
            threshold = percentile_cap

        # -------------------------
        # 1. Бинаризация с нижней и верхней границей (глобальный критерий)
        # -------------------------
        effective_threshold = max(int(threshold), 27)
        
        global_mask = cv2.inRange(
            self.gray,
            min_brightness,
            effective_threshold
        )
        global_percent = 100 * cv2.countNonZero(global_mask) / global_mask.size

        # -------------------------
        # 1б. Локальный критерий (пятна темнее своего окружения)
        # -------------------------
        mask = global_mask
        local_mask = np.zeros_like(global_mask)

        if use_local_contrast and global_percent < max_talc_percent:
            k_size = local_kernel_size if local_kernel_size % 2 == 1 else local_kernel_size + 1

            local_background = cv2.GaussianBlur(
                self.gray, (k_size, k_size), 0
            ).astype(np.int16)
            local_diff = local_background - self.gray.astype(np.int16)

            non_ore_area_mask = np.where(
                labels.reshape(self.gray.shape) == non_ore_label, 255, 0
            ).astype(np.uint8)
            non_ore_ceiling_mask = cv2.inRange(self.gray, min_brightness, int(non_ore_mean))

            current_margin = local_contrast_margin
            for attempt in range(8):
                local_mask = np.where(
                    local_diff >= current_margin, 255, 0
                ).astype(np.uint8)

                local_mask = cv2.bitwise_and(local_mask, non_ore_area_mask)
                local_mask = cv2.bitwise_and(local_mask, non_ore_ceiling_mask)

                combined = cv2.bitwise_or(global_mask, local_mask)
                combined_percent = 100 * cv2.countNonZero(combined) / combined.size

                if combined_percent <= max_talc_percent:
                    break
                current_margin += 5

            mask = cv2.bitwise_or(global_mask, local_mask)

        # --- ЖЕСТКАЯ ВРЕЗКА ТУТ (В САМОМ НАЧАЛЕ) ---
        # Подмешиваем пиксели < 25 до фильтров, чтобы они сглаживались вместе со всеми
        force_dark_mask = cv2.inRange(self.gray, min_brightness, 25)
        mask = cv2.bitwise_or(mask, force_dark_mask)
        # -------------------------------------------

        # -------------------------
        # 2. Убираем шум (мелкие одиночные пиксели)
        # -------------------------
        small_kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)

        if cv2.countNonZero(opened) > 0 or cv2.countNonZero(mask) == 0:
            mask = opened

        # -------------------------
        # 3. Смыкание близких пятен в одну область
        # -------------------------
        close_size = max(3, merge_distance)
        if close_size % 2 == 0:
            close_size += 1

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        final_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

        # -------------------------
        # 4. Заливка только МЕЛКИХ внутренних "дыр"
        # -------------------------
        h, w = final_mask.shape
        flood_filled = final_mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood_filled)

        num_labels, labels_holes, stats, _ = cv2.connectedComponentsWithStats(holes)

        small_holes = np.zeros_like(final_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] <= max_hole_area:
                small_holes[labels_holes == i] = 255

        final_mask = cv2.bitwise_or(final_mask, small_holes)

        # -------------------------
        # 5. Сглаживание и контролируемое УМЕНЬШЕНИЕ (сокращение размера)
        # -------------------------
        s_size = smooth_size if smooth_size % 2 == 1 else smooth_size + 1
        s_size = max(3, s_size)

        before_smooth = final_mask.copy()

        blurred = cv2.GaussianBlur(final_mask, (s_size, s_size), 0)
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # --- КОРРЕКЦИЯ РАЗДУВАНИЯ ---
        # Принудительное легкое сужение (эрозия), чтобы убрать лишний объем от размытия
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        smoothed = cv2.erode(smoothed, shrink_kernel, iterations=1)
        # -----------------------------

        num_labels, labels_before, stats, _ = cv2.connectedComponentsWithStats(before_smooth)
        for i in range(1, num_labels):
            comp_mask = (labels_before == i)
            if cv2.countNonZero((smoothed > 0) & comp_mask) == 0:
                smoothed[comp_mask] = 255

        if cv2.countNonZero(smoothed) > 0 or cv2.countNonZero(final_mask) == 0:
            final_mask = smoothed

        # -------------------------
        # 6. Отсев крупных ОДНОРОДНЫХ пятен (вероятно, не тальк)
        # -------------------------
        if max_component_percent is not None:
            max_component_area = final_mask.size * max_component_percent / 100
            num_labels, labels_big, stats, _ = cv2.connectedComponentsWithStats(final_mask)

            component_infos = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                area_pct = 100 * area / final_mask.size
                if area_pct >= 0.1:
                    comp_std = float(self.gray[labels_big == i].std())
                    component_infos.append((area_pct, comp_std, i))

            for area_pct, comp_std, i in component_infos:
                if area_pct * final_mask.size / 100 <= max_component_area:
                    continue

                comp_mask = (labels_big == i)
                if comp_std < homogeneity_std_threshold:
                    final_mask[comp_mask] = 0

        # -------------------------
        # 7. Отсев мелких отдельных областей
        # -------------------------
        if min_area > 0:
            num_labels, labels_final, stats, _ = cv2.connectedComponentsWithStats(final_mask)

            filtered_mask = np.zeros_like(final_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_area:
                    filtered_mask[labels_final == i] = 255

            final_mask = filtered_mask

        # !!! СТАРЫЙ БЛОК ИЗ ХВОСТА УДАЛЕН !!!
        # Теперь маска не раздувается пикселями в самом конце.

        final_percent = 100 * cv2.countNonZero(final_mask) / final_mask.size
        print(f"Итоговая площадь талька: {final_percent:.2f}%")

        self.masks["talc"] = final_mask
        return final_mask

    # ------------------------
    # Тонкие срастания
    # ------------------------
    def create_fine_mask(self, threshold=135, min_area=500):
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
            27,
            135
        )

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