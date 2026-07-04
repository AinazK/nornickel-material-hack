import sys
import os
import io
import time
import base64
import streamlit as st

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from utils.ainaz.imageLoader import load_image
from ore_processor import OreProcessor

# ---------------------------------------------------------------
# Patch для совместимости streamlit-drawable-canvas с новыми версиями
# Streamlit (>=1.31), где была удалена/переименована внутренняя функция
# streamlit.elements.image.image_to_url, от которой зависит canvas.
# ---------------------------------------------------------------
import streamlit.elements.image as _st_image_module

if not hasattr(_st_image_module, "image_to_url"):
    def _image_to_url(image, width=None, clamp=False, channels="RGB",
                       output_format="auto", image_id=None, allow_emoji=False):
        if isinstance(image, Image.Image):
            pil_img = image
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.open(image)

        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")

        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        data = buffer.getvalue()

        # Предпочтительный путь: зарегистрировать картинку в родном
        # MediaFileManager Streamlit и получить настоящую ссылку /media/...
        # — именно так это делает сам st.image в актуальных версиях.
        try:
            import streamlit.runtime as _st_runtime
            _media_mgr = _st_runtime.get_instance().media_file_mgr
            return _media_mgr.add(
                data,
                "image/png",
                str(image_id) if image_id else "drawable-canvas-bg",
            )
        except Exception:
            # Фолбэк на data-URI, если Runtime недоступен
            # (например, скрипт запущен не через `streamlit run`).
            encoded = base64.b64encode(data).decode()
            return f"data:image/png;base64,{encoded}"

    _st_image_module.image_to_url = _image_to_url

from streamlit_drawable_canvas import st_canvas

st.set_page_config(
    page_title="Анализ руды: Алгоритм vs Ручная разметка",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #00cc99, #0099ff);
        color: white;
        font-size: 1.1rem;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Анализ руды: Алгоритм vs Ручная разметка")
st.caption("Сравнение автоматической сегментации с экспертной (ручной) разметкой оператора")

# -----------------------------
# Цвета и подписи классов (единые для алгоритма и ручной разметки)
# -----------------------------
COLORS_RGB = {
    "talc": [0, 0, 255],      # Синий
    "fine": [255, 0, 0],      # Красный
    "normal": [0, 255, 0],    # Зелёный
}
COLORS_CSS = {
    "talc": "rgb(0, 0, 255)",
    "fine": "rgb(255, 0, 0)",
    "normal": "rgb(0, 255, 0)",
}
CLASS_LABELS = {
    "talc": "🔵 Тальк",
    "fine": "🔴 Тонкие срастания",
    "normal": "🟢 Крупные области",
    "waste": "⚪ Пустая порода / Остальное",
}

# -----------------------------
# 1. Загрузка файла
# -----------------------------
file = st.file_uploader(
    "Выберите изображение шлифа",
    type=["png", "jpg", "jpeg", "tiff"]
)

if file is None:
    st.info("Загрузите изображение для начала анализа")
    st.stop()

st.success("Файл загружен")

# Сбрасываем состояние разметки при смене файла
if st.session_state.get("_current_file") != file.name:
    st.session_state["_current_file"] = file.name
    st.session_state["_canvas_key_suffix"] = 0
    st.session_state.pop("manual_stats", None)
    st.session_state.pop("algo_stats", None)

# -----------------------------
# 2. Парсинг изображения и инициализация процессора
# -----------------------------
img_data = load_image(file)

file.seek(0)
file_bytes = file.read()

# Оригинальное изображение (то, что будет фоном для ручной разметки)
original_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")

processor = OreProcessor()
processor.load_from_bytes(file_bytes)
processor.apply_clahe_preprocessing(target_brightness=120)

# -----------------------------
# 3. Метаданные
# -----------------------------
with st.expander("📊 Метаданные изображения", expanded=False):
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.metric("Файл", img_data.filename)
        st.metric("Формат", img_data.image_format)
        st.metric("Размер файла (байт)", img_data.file_size)
    with meta_col2:
        st.metric("Ширина (px)", img_data.width)
        st.metric("Высота (px)", img_data.height)
        st.metric("Каналы", img_data.image.shape[2] if len(img_data.image.shape) == 3 else 1)

st.divider()

# -----------------------------
# 4. Автоматический анализ (алгоритм)
# -----------------------------
st.header("1️⃣ Автоматический анализ (алгоритм)")

if st.button("🔬 Запустить автоматический анализ", type="primary", use_container_width=True):
    with st.spinner("Выполняется математический анализ масок минералов..."):
        processor.create_talc_mask()
        processor.create_fine_mask()
        processor.create_normal_mask()

        stats = processor.get_statistics()
        ore_class = processor.classify_ore(stats)

        time.sleep(0.3)

        st.session_state["algo_stats"] = stats
        st.session_state["ore_class"] = ore_class
        st.session_state["overlay_all"] = processor.get_combined_overlay()
        st.session_state["overlay_rgb"] = processor.rgb.copy()
        st.session_state["mask_talc"] = processor.masks["talc"].copy() if processor.masks["talc"] is not None else None
        st.session_state["mask_fine"] = processor.masks["fine"].copy() if processor.masks["fine"] is not None else None
        st.session_state["mask_normal"] = processor.masks["normal"].copy() if processor.masks["normal"] is not None else None

if "algo_stats" in st.session_state:
    stats = st.session_state["algo_stats"]
    ore_class = st.session_state["ore_class"]

    st.success(f"✅ Анализ завершён! Тип породы: **{ore_class}**")

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(CLASS_LABELS["talc"], f"{stats['talc']}%")
    m_col2.metric(CLASS_LABELS["fine"], f"{stats['fine']}%")
    m_col3.metric(CLASS_LABELS["normal"], f"{stats['normal']}%")
    m_col4.metric(CLASS_LABELS["waste"], f"{stats['waste']}%")

    st.subheader("🎨 Маски алгоритма по отдельности")
    algo_tab_all, algo_tab_talc, algo_tab_fine, algo_tab_normal = st.tabs([
        "🌈 Все маски вместе",
        CLASS_LABELS["talc"],
        CLASS_LABELS["fine"],
        CLASS_LABELS["normal"],
    ])

    IMG_DISPLAY_WIDTH = 600
    base_rgb = st.session_state["overlay_rgb"]

    with algo_tab_all:
        st.image(st.session_state["overlay_all"], caption="Совмещённая карта сегментации", width=IMG_DISPLAY_WIDTH)

    with algo_tab_talc:
        mask = st.session_state["mask_talc"]
        if mask is not None:
            overlay = base_rgb.copy()
            overlay[mask > 0] = COLORS_RGB["talc"]
            st.image(overlay, caption="Только Тальк (синий)", width=IMG_DISPLAY_WIDTH)
        else:
            st.warning("Маска Талька пуста")

    with algo_tab_fine:
        mask = st.session_state["mask_fine"]
        if mask is not None:
            overlay = base_rgb.copy()
            overlay[mask > 0] = COLORS_RGB["fine"]
            st.image(overlay, caption="Только Тонкие срастания (красный)", width=IMG_DISPLAY_WIDTH)
        else:
            st.warning("Маска тонких срастаний пуста")

    with algo_tab_normal:
        mask = st.session_state["mask_normal"]
        if mask is not None:
            overlay = base_rgb.copy()
            overlay[mask > 0] = COLORS_RGB["normal"]
            st.image(overlay, caption="Только Крупные области (зелёный)", width=IMG_DISPLAY_WIDTH)
        else:
            st.warning("Маска крупных областей пуста")
else:
    st.info("Нажмите кнопку выше, чтобы получить результаты алгоритма")

st.divider()

# -----------------------------
# 5. Ручная разметка (эксперт) — один холст с переключением классов
# -----------------------------
st.header("2️⃣ Ручная разметка (ваш выбор)")
st.caption("Выберите нужный класс в выпадающем списке, настройте кисть и закрашивайте области прямо на изображении.")

DISPLAY_WIDTH = 600
w, h = original_pil.size
scale = DISPLAY_WIDTH / w
canvas_height = int(h * scale)
pil_img_resized = original_pil.resize((DISPLAY_WIDTH, canvas_height))

# Элементы управления разметкой
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])

with ctrl_col1:
    # Выбор активного класса для рисования
    active_label = st.selectbox(
        "Что вы хотите разметить сейчас?",
        options=[CLASS_LABELS["talc"], CLASS_LABELS["fine"], CLASS_LABELS["normal"]],
        key="active_class_selector"
    )
    
    # Определяем внутреннее имя класса и цвет кисти на основе выбора
    if active_label == CLASS_LABELS["talc"]:
        current_class = "talc"
    elif active_label == CLASS_LABELS["fine"]:
        current_class = "fine"
    else:
        current_class = "normal"
        
    chosen_color = COLORS_CSS[current_class]

with ctrl_col2:
    brush_size = st.slider("Размер кисти", 3, 40, 12)

with ctrl_col3:
    st.write("") # Отступ для выравнивания кнопки
    st.write("") 
    if st.button("🗑️ Очистить разметку", use_container_width=True):
        st.session_state["_canvas_key_suffix"] = st.session_state.get("_canvas_key_suffix", 0) + 1
        st.session_state.pop("manual_stats", None)

suffix = st.session_state.get("_canvas_key_suffix", 0)
file_id = st.session_state.get("_current_file", "default")

# Один стабильный холст для всех классов
st.info(f"🎨 Сейчас активна кисть: **{active_label}**")
canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=brush_size,
    stroke_color=chosen_color,
    background_image=pil_img_resized,
    update_streamlit=True,
    height=canvas_height,
    width=DISPLAY_WIDTH,
    drawing_mode="freedraw",
    key=f"single_canvas_{file_id}_{suffix}",
)

def compute_manual_stats_single(canvas_data):
    """Считает % площади по каждому классу на основе одного холста,
    анализируя RGB-цвета нарисованных линий."""
    if canvas_data is None or canvas_data.image_data is None:
        return None

    # Получаем RGBA матрицу (высота x ширина x 4)
    img = canvas_data.image_data
    total_pixels = img.shape[0] * img.shape[1]
    
    # Извлекаем маски по цветам (проверяем альфу > 0 и соответствие RGB)
    alpha = img[:, :, 3] > 0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    
    # Сравниваем с допусками, так как антиалиасинг canvas может слегка менять цвета на границах
    # Синий (Talc): B > 200, R < 50, G < 50
    talc_mask = alpha & (b > 200) & (r < 50) & (g < 50)
    # Красный (Fine): R > 200, G < 50, B < 50
    fine_mask = alpha & (r > 200) & (g < 50) & (b < 50)
    # Зеленый (Normal): G > 200, R < 50, B < 50
    normal_mask = alpha & (g > 200) & (r < 50) & (b < 50)
    
    stats = {
        "talc": round(100 * talc_mask.sum() / total_pixels, 2),
        "fine": round(100 * fine_mask.sum() / total_pixels, 2),
        "normal": round(100 * normal_mask.sum() / total_pixels, 2),
    }
    
    stats["waste"] = round(max(0.0, 100 - sum(stats[k] for k in ["talc", "fine", "normal"])), 2)
    return stats


if st.button("✅ Зафиксировать ручную разметку", use_container_width=True):
    result = compute_manual_stats_single(canvas_result)
    if result is not None and (result["talc"] > 0 or result["fine"] > 0 or result["normal"] > 0):
        st.session_state["manual_stats"] = result
    else:
        st.warning("Сначала закрасьте хотя бы одну область на рисунке!")

if "manual_stats" in st.session_state:
    mstats = st.session_state["manual_stats"]
    mm_col1, mm_col2, mm_col3, mm_col4 = st.columns(4)
    mm_col1.metric(CLASS_LABELS["talc"], f"{mstats['talc']}%")
    mm_col2.metric(CLASS_LABELS["fine"], f"{mstats['fine']}%")
    mm_col3.metric(CLASS_LABELS["normal"], f"{mstats['normal']}%")
    mm_col4.metric(CLASS_LABELS["waste"], f"{mstats['waste']}%")
else:
    st.info("Разметьте минералы на холсте и нажмите «Зафиксировать ручную разметку»")

# -----------------------------
# 6. Сравнение
# -----------------------------
st.header("3️⃣ Сравнение алгоритма и ручной разметки")

if "algo_stats" in st.session_state and "manual_stats" in st.session_state:
    algo = st.session_state["algo_stats"]
    manual = st.session_state["manual_stats"]
    
    # Извлекаем данные холста для расчета пространственных метрик
    # (canvas_result должен быть доступен из предыдущего шага)
    img_canvas = canvas_result.image_data if (canvas_result is not None and canvas_result.image_data is not None) else None

    rows = []
    diffs = []
    
    # Словари для хранения IoU и Dice по каждому классу
    class_iou = {}
    class_dice = {}

    # Подготавливаем маски для попиксельного сравнения IoU / Dice
    if img_canvas is not None:
        import cv2  # OpenCV обычно предустановлен, либо используется для ресайза масок алгоритма
        
        # Размеры холста
        canvas_h, canvas_w = img_canvas.shape[0], img_canvas.shape[1]
        
        # 1. Извлекаем бинарные маски эксперта из холста по цветам
        alpha = img_canvas[:, :, 3] > 0
        r_c, g_c, b_c = img_canvas[:, :, 0], img_canvas[:, :, 1], img_canvas[:, :, 2]
        
        manual_masks_bin = {
            "talc": alpha & (b_c > 200) & (r_c < 50) & (g_c < 50),
            "fine": alpha & (r_c > 200) & (g_c < 50) & (b_c < 50),
            "normal": alpha & (g_c > 200) & (r_c < 50) & (b_c < 50),
        }
        # Маска "waste" для эксперта — всё, что НЕ закрашено
        manual_masks_bin["waste"] = ~alpha

        # 2. Извлекаем маски алгоритма и сжимаем их до размера холста
        algo_masks_bin = {}
        for key in ["talc", "fine", "normal"]:
            orig_mask = st.session_state.get(f"mask_{key}")
            if orig_mask is not None:
                # Приводим к размеру холста, используя метод ближайшего соседа INTER_NEAREST для бинарных масок
                resized_mask = cv2.resize(orig_mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
                algo_masks_bin[key] = resized_mask > 0
            else:
                algo_masks_bin[key] = np.zeros((canvas_h, canvas_w), dtype=bool)
                
        # Маска "waste" для алгоритма — то, где нет ни одного из трех классов
        algo_masks_bin["waste"] = ~(algo_masks_bin["talc"] | algo_masks_bin["fine"] | algo_masks_bin["normal"])

        # 3. Считаем IoU и Dice для каждого класса
        for key in ["talc", "fine", "normal", "waste"]:
            m_mask = manual_masks_bin[key]
            a_mask = algo_masks_bin[key]
            
            intersection = np.logical_and(m_mask, a_mask).sum()
            union = np.logical_or(m_mask, a_mask).sum()
            
            # Расчет IoU (Жаккар)
            if union == 0:
                iou = 100.0 if intersection == 0 else 0.0
            else:
                iou = (intersection / union) * 100
                
            # Расчет Dice Coefficient
            total_elements = m_mask.sum() + a_mask.sum()
            if total_elements == 0:
                dice = 100.0 if intersection == 0 else 0.0
            else:
                dice = ((2.0 * intersection) / total_elements) * 100
                
            class_iou[key] = round(iou, 1)
            class_dice[key] = round(dice, 1)
    else:
        # Фолбэк, если данные холста почему-то недоступны
        for key in ["talc", "fine", "normal", "waste"]:
            class_iou[key] = 0.0
            class_dice[key] = 0.0

    # Заполняем строки для расширенной таблицы
    for key in ["talc", "fine", "normal", "waste"]:
        a = float(algo[key])
        m = float(manual[key])
        diff = round(a - m, 2)
        diffs.append(abs(diff))
        
        rows.append({
            "Класс": CLASS_LABELS[key],
            "Алгоритм (%)": a,
            "Ручная разметка (%)": m,
            "Разница (п.п.)": diff,
            "Геометрическое совпадение (IoU, %)": class_iou[key],
            "Точность контуров (Dice, %)": class_dice[key],
        })

    st.subheader("📊 Экспертная таблица пространственного совпадения")
    st.caption("Метрики IoU и Dice оценивают не просто площадь, а то, насколько точно совпали контуры и координаты закрашенных областей.")
    
    # Настраиваем отображение колонок (добавляем цветной прогресс-бар для IoU и Dice прямо в ячейки!)
    st.data_editor(
        rows,
        column_config={
            "Разница (п.п.)": st.column_config.NumberColumn(format="%+.2f"),
            "Геометрическое совпадение (IoU, %)": st.column_config.ProgressColumn(
                "Пересечение (IoU, %)",
                help="Intersection over Union. 100% — идеальное пространственное совпадение.",
                min_value=0, max_value=100, format="%.1f%%"
            ),
            "Точность контуров (Dice, %)": st.column_config.ProgressColumn(
                "Сходство контуров (Dice, %)",
                help="Dice Coefficient. Оценивает гармоническое сходство пересечения границ масок.",
                min_value=0, max_value=100, format="%.1f%%"
            )
        },
        use_container_width=True,
        hide_index=True,
        disabled=True
    )

    # Среднее пространственное сходство на основе Dice
    mean_spatial_similarity = sum(class_dice.values()) / len(class_dice)
    similarity = round(mean_spatial_similarity, 1)

    st.divider()

    # 2. Вывод итогового вердикта
    st.subheader("🎯 Итоговый вердикт пространственного сходства")
    
    if similarity >= 80:
        bar_color = "#00cc99"  # Зеленый
        status_text = "Отличная точность! Алгоритм практически идеально угадал координаты и форму ваших масок."
        alert_type = st.success
    elif similarity >= 50:
        bar_color = "#ffcc00"  # Желтый
        status_text = "Умеренное пространственное сходство. Площади похожи, но границы и расположение объектов частично расходятся."
        alert_type = st.warning
    else:
        bar_color = "#ff3333"  # Красный
        status_text = "Низкое совпадение контуров. Проверьте правильность разметки или скорректируйте пороговые фильтры алгоритма."
        alert_type = st.error

    st.markdown(f"""
    <div style="
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 8px solid {bar_color};
        margin-bottom: 20px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 1.2rem; font-weight: bold; color: #31333F;">Интегральный индекс сходства контуров (Dice Index):</span>
            <span style="font-size: 2rem; font-weight: bold; color: {bar_color};">{similarity}%</span>
        </div>
        <div style="
            background-color: #e0e0e0; 
            border-radius: 5px; 
            height: 15px; 
            width: 100%; 
            margin-top: 10px;
            overflow: hidden;
        ">
            <div style="
                background-color: {bar_color}; 
                width: {similarity}%; 
                height: 100%; 
                border-radius: 5px;
                transition: width 0.5s ease-in-out;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    alert_type(status_text)

else:
    st.info("Чтобы увидеть пространственное сравнение масок, выполните автоматический анализ и ручную разметку на холсте.")