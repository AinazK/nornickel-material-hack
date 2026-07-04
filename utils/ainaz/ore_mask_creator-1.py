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
    def create_talc_mask(
        self,
        outlier_std_margin=1.2,
        max_talc_percent=30,
        min_brightness=0,
        use_local_contrast=True,
        local_kernel_size=51,
        local_contrast_margin=30,
        merge_distance=25,
        max_hole_area=500,
        smooth_size=27,
        max_component_percent=2,
        homogeneity_std_threshold=10,
        min_area=0
    ):
        """
        Определяем два основных материала на фото — рудную и нерудную
        область — через K-means кластеризацию по яркости (2 кластера).
        Более тёмный кластер считаем нерудной областью (вмещающая порода,
        обычно ниже отражательная способность), более светлый — рудной
        (рудные минералы обычно ярче под отражённым светом).

        Тальк ищем как пиксели, которые статистически значимо темнее
        нерудной области — то есть выходят за пределы её обычного разброса
        яркости (среднее минус outlier_std_margin стандартных отклонений).
        Если на фото нет такой отдельной, ещё более тёмной популяции
        пикселей — тальк не найден, маска остаётся пустой.

        Дополнительно порог всегда ограничен потолком max_talc_percent:
        даже если статистический критерий даёт больше — берём тот порог,
        что жёстче (используя перцентиль яркости всего изображения). Это
        подстраховка от переразрастания талька на изображениях, где
        нерудная область имеет широкий разброс яркости.

        outlier_std_margin — насколько стандартных отклонений ниже среднего
        нерудной области должен быть пиксель, чтобы считаться тальком.
        Чем больше значение — тем строже критерий.

        max_talc_percent — жёсткий потолок площади талька в процентах от
        всего изображения (по умолчанию 20%). Порог никогда не даст больше
        этого процента "сырых" тальковых пикселей до всех дальнейших
        морфологических операций.

        min_brightness — нижняя граница диапазона талька. Пиксели темнее
        этого значения (артефакты, царапины, дефекты съёмки) в тальк НЕ
        включаются. По умолчанию 0 — нижней границы нет.

        use_local_contrast — включает второй, локальный критерий поиска
        талька в дополнение к глобальному. Ловит пятна, которые по
        абсолютной яркости попадают в обычный разброс нерудной области
        (и поэтому не проходят глобальный порог), но заметно темнее своего
        непосредственного окружения. Критерий применяется ТОЛЬКО внутри
        нерудной области — рудную область не трогает, чтобы не плодить
        ложные срабатывания на её границах.

        local_kernel_size — размер окна (в пикселях) для расчёта локальной
        фоновой яркости вокруг каждого пикселя (через размытие). Должен
        быть заметно больше типичного пятна талька, чтобы фон вокруг
        пятна считался корректно, но не настолько большим, чтобы захватить
        соседние структуры.

        local_contrast_margin — на сколько единиц яркости пиксель должен
        быть темнее своего локального фона, чтобы считаться тальком по
        локальному критерию. Чем меньше значение — тем чувствительнее
        (больше слабоконтрастных пятен ловится, но и больше риск шума).

        merge_distance — расстояние (в пикселях) между близкими пятнами
        талька, которое нужно "замостить", чтобы они слились в одну
        область. Реализовано через морфологическое смыкание (dilate ->
        erode) с эллиптическим ядром соответствующего размера. Это
        единственный шаг, отвечающий за объединение пятен — не путать со
        сглаживанием ниже.

        max_hole_area — максимальная площадь (в пикселях) внутренней дыры,
        которую нужно закрасить. Дыры меньше или равные этому значению
        закрашиваются, более крупные остаются нетронутыми.

        smooth_size — размер ядра сглаживания границ (GaussianBlur), в
        пикселях, задаётся НАПРЯМУЮ и не зависит от merge_distance. Раньше
        оно вычислялось как merge_distance * smooth_strength, из-за чего
        при большом merge_distance сглаживание превращалось во второй,
        даже более сильный шаг слияния — отсюда и раздувание областей.
        Теперь это чисто про сглаживание зубчатых краёв, небольшое
        значение (10-20 px), не растягивающее области дальше, чем уже
        сделал merge_distance.

        max_component_percent — если одна связная область талька занимает
        больше этого процента от всего изображения (по умолчанию 2%) И
        при этом однородна по яркости внутри (см. homogeneity_std_threshold)
        — считаем, что это не тальк, а крупный одиночный тёмный минерал,
        тень или артефакт съёмки, и вырезаем её из маски целиком. Реальный
        тальк на фото обычно более текстурный/неоднородный внутри, даже
        если занимает большую площадь.

        homogeneity_std_threshold — порог стандартного отклонения яркости
        внутри области (в исходном изображении, не в маске), ниже которого
        область считается "однородной" (плоское пятно без текстуры) и
        может быть вырезана, если она к тому же крупнее max_component_percent.
        Чем меньше значение — тем строже критерий однородности (области
        должны быть совсем ровными по цвету, чтобы попасть под вырезание).

        min_area — минимальная площадь (в пикселях) итоговой отдельной
        области талька, которую нужно оставить. Компоненты меньше этого
        значения удаляются из маски как шум. По умолчанию 0 — не убирается.
        """

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

        # Более тёмный кластер = нерудная область, более светлый = рудная
        if centers[0] < centers[1]:
            non_ore_label, ore_label = 0, 1
        else:
            non_ore_label, ore_label = 1, 0

        non_ore_pixels = pixels[labels == non_ore_label]
        non_ore_mean = float(non_ore_pixels.mean())
        non_ore_std = float(non_ore_pixels.std())

        threshold = non_ore_mean - outlier_std_margin * non_ore_std

        print(
            f"Нерудная область: центр={centers[non_ore_label]:.1f}, "
            f"std={non_ore_std:.1f}"
        )
        print(f"Рудная область: центр={centers[ore_label]:.1f}")
        print(
            f"Порог талька по std (нерудная - {outlier_std_margin} * std): "
            f"{threshold:.1f}"
        )

        # -------------------------
        # Потолок: тальк не должен занимать больше max_talc_percent
        # -------------------------
        percentile_cap = float(np.percentile(self.gray, max_talc_percent))
        stat_threshold = threshold
        if threshold > percentile_cap:
            threshold = percentile_cap
            print(
                f"⚠ Сработал потолок {max_talc_percent}%: статистический "
                f"порог ({stat_threshold:.1f}) был выше потолка "
                f"({percentile_cap:.1f}) — часть талька могла обрезаться. "
                f"Если тальк пропускается — поднимите max_talc_percent."
            )
        else:
            print(
                f"✓ Сработал статистический порог ({stat_threshold:.1f}), "
                f"потолок {max_talc_percent}% ({percentile_cap:.1f}) не "
                f"ограничивал. Если тальк всё ещё пропускается — снижайте "
                f"outlier_std_margin (сейчас {outlier_std_margin})."
            )

        if threshold <= self.gray.min():
            print("Тальк не найден: на фото нет пикселей темнее нерудной области.")

        print(f"Итоговый порог талька: {threshold:.1f}")

        # -------------------------
        # 1. Бинаризация с нижней и верхней границей (глобальный критерий)
        # -------------------------
        global_mask = cv2.inRange(
            self.gray,
            min_brightness,
            max(int(threshold), 0)
        )

        global_percent = 100 * cv2.countNonZero(global_mask) / global_mask.size
        print(f"Глобальный критерий: {global_percent:.2f}% пикселей")

        # -------------------------
        # 1б. Локальный критерий (пятна темнее своего окружения)
        # -------------------------
        # Ищем итоговый local_contrast_margin так, чтобы комбинированная
        # маска (глобальный + локальный) тоже не превышала max_talc_percent.
        # Если рудный порог сам по себе уже упёрся в потолок — локальный
        # критерий вообще не добавляем, добавлять уже некуда.
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

                # Ограничиваем локальный критерий только нерудной областью и
                # яркостью не выше нерудного среднего — чтобы не цеплять
                # границы рудной области или случайные локальные перепады.
                local_mask = cv2.bitwise_and(local_mask, non_ore_area_mask)
                local_mask = cv2.bitwise_and(local_mask, non_ore_ceiling_mask)

                combined = cv2.bitwise_or(global_mask, local_mask)
                combined_percent = 100 * cv2.countNonZero(combined) / combined.size

                if combined_percent <= max_talc_percent:
                    break

                # Слишком много — ужесточаем локальный критерий и пробуем снова
                current_margin += 5
            else:
                print(
                    f"⚠ Локальный критерий не удалось уложить в потолок "
                    f"{max_talc_percent}% даже при margin={current_margin} — "
                    f"взят самый строгий вариант из перебора."
                )

            if current_margin != local_contrast_margin:
                print(
                    f"Локальный критерий ужесточён, чтобы уложиться в потолок "
                    f"{max_talc_percent}%: local_contrast_margin "
                    f"{local_contrast_margin} -> {current_margin}"
                )

            local_percent = 100 * cv2.countNonZero(local_mask) / local_mask.size
            print(f"Локальный критерий (темнее фона на {current_margin}+): "
                  f"{local_percent:.2f}% пикселей")

            mask = cv2.bitwise_or(global_mask, local_mask)
        elif use_local_contrast:
            print(
                f"Глобальный критерий уже на потолке ({global_percent:.2f}% "
                f">= {max_talc_percent}%) — локальный критерий пропущен."
            )

        raw_percent = 100 * cv2.countNonZero(mask) / mask.size
        print(f"Сырая маска до обработки (глобальный + локальный): "
              f"{raw_percent:.2f}% пикселей")

        # -------------------------
        # 2. Убираем шум (мелкие одиночные пиксели)
        # -------------------------
        small_kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)

        # Защита: если шумоподавление стёрло непустую маску полностью —
        # откатываемся и пропускаем этот шаг, а не теряем весь тальк.
        if cv2.countNonZero(opened) > 0 or cv2.countNonZero(mask) == 0:
            mask = opened
        else:
            print("Шумоподавление обнулило маску — шаг пропущен.")

        # -------------------------
        # 3. Смыкание близких пятен в одну область
        # -------------------------
        # Единственный шаг, отвечающий за слияние пятен. Ядро смыкания
        # примерно равно расстоянию, которое нужно "перекрыть" между
        # соседними пятнами.
        close_size = max(3, merge_distance)
        if close_size % 2 == 0:
            close_size += 1  # нечётный размер ядра

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_size, close_size)
        )

        final_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1
        )

        merged_percent = 100 * cv2.countNonZero(final_mask) / final_mask.size
        print(f"После слияния (merge_distance={merge_distance}): "
              f"{merged_percent:.2f}% пикселей")

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
        # 5. Сглаживание границ (чисто косметическое, без слияния)
        # -------------------------
        # Небольшое, фиксированное ядро — только убирает зубцы по краям
        # уже сформированных областей, не мостит новые промежутки.
        s_size = smooth_size if smooth_size % 2 == 1 else smooth_size + 1
        s_size = max(3, s_size)

        before_smooth = final_mask.copy()

        blurred = cv2.GaussianBlur(final_mask, (s_size, s_size), 0)
        _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Защита: мелкие изолированные пятна (меньше половины ядра
        # сглаживания) блюр стирает целиком. Находим каждый компонент
        # исходной маски и, если после сглаживания от него не осталось
        # вообще ничего — возвращаем его как есть (без сглаживания), а не
        # теряем полностью. Крупные области при этом остаются сглаженными.
        num_labels, labels_before, stats, _ = cv2.connectedComponentsWithStats(before_smooth)
        restored_count = 0
        for i in range(1, num_labels):
            comp_mask = (labels_before == i)
            if cv2.countNonZero((smoothed > 0) & comp_mask) == 0:
                smoothed[comp_mask] = 255
                restored_count += 1

        if restored_count > 0:
            print(f"Сглаживание стирало {restored_count} мелких пятен — "
                  f"восстановлены без сглаживания.")

        if cv2.countNonZero(smoothed) > 0 or cv2.countNonZero(final_mask) == 0:
            final_mask = smoothed
        else:
            print("Сглаживание обнулило маску — шаг пропущен.")

        # -------------------------
        # 6. Отсев крупных ОДНОРОДНЫХ пятен (вероятно, не тальк)
        # -------------------------
        # Настоящий тальк, даже занимая большую площадь, обычно неоднороден
        # по текстуре. Если одно связное пятно одновременно (а) крупнее
        # max_component_percent от всего изображения и (б) очень ровное по
        # яркости внутри (std ниже homogeneity_std_threshold) — это, скорее
        # всего, отдельный крупный тёмный минерал, тень или артефакт съёмки,
        # а не тальк. Вырезаем такие пятна целиком.
        if max_component_percent is not None:
            max_component_area = final_mask.size * max_component_percent / 100
            num_labels, labels_big, stats, _ = cv2.connectedComponentsWithStats(final_mask)

            # Диагностика: показываем размеры всех заметных компонент (от
            # 0.1% площади фото), даже если они не проходят порог — чтобы
            # было видно, какой процент реально выставлять в
            # max_component_percent для конкретного фото.
            component_infos = []
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                area_pct = 100 * area / final_mask.size
                if area_pct >= 0.1:
                    comp_std = float(self.gray[labels_big == i].std())
                    component_infos.append((area_pct, comp_std, i))

            component_infos.sort(reverse=True)
            if component_infos:
                print("Заметные связные пятна (площадь% / std внутри):")
                for area_pct, comp_std, _ in component_infos[:10]:
                    print(f"  {area_pct:.2f}% / std={comp_std:.1f}")

            for area_pct, comp_std, i in component_infos:
                if area_pct * final_mask.size / 100 <= max_component_area:
                    continue

                comp_mask = (labels_big == i)

                if comp_std < homogeneity_std_threshold:
                    final_mask[comp_mask] = 0
                    print(
                        f"Вырезано крупное однородное пятно (не тальк): "
                        f"площадь={area_pct:.2f}%, "
                        f"std внутри={comp_std:.1f} (< {homogeneity_std_threshold})"
                    )
                else:
                    print(
                        f"Крупное пятно оставлено (текстурное, похоже на "
                        f"реальный тальк): площадь={area_pct:.2f}%, "
                        f"std внутри={comp_std:.1f}"
                    )

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

        final_percent = 100 * cv2.countNonZero(final_mask) / final_mask.size
        print(f"Итоговая площадь талька: {final_percent:.2f}%")

        self.masks["talc"] = final_mask
        return final_mask

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

    creator.create_talc_mask(
        outlier_std_margin=1.2,
        max_talc_percent=30,
        min_brightness=0,
        use_local_contrast=True,
        local_kernel_size=51,
        local_contrast_margin=30,
        merge_distance=25,
        max_hole_area=500,
        smooth_size=30,
        max_component_percent=2,
        homogeneity_std_threshold=10,
        min_area=300
    )
    creator.create_fine_mask()
    creator.create_normal_mask()

    creator.show()

    creator.save("masks")

    # Вычисление процентного соотношения
    total_pixels = creator.image.shape[0] * creator.image.shape[1]
    print("\nПроцентное соотношение масок (от общей площади изображения):")
    for name, mask in creator.masks.items():
        if mask is not None:
            count = np.count_nonzero(mask)
            percent = (count / total_pixels) * 100
            print(f"  {name}: {percent:.2f}%")