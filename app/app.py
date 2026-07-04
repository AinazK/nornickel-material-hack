import sys
import os
import time
import streamlit as st

import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

from utils.ainaz.imageLoader import load_image
# Импортируем наш новый коннектор
from ore_processor import OreProcessor

st.set_page_config(
    page_title="Анализ руды (MVP)",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #00cc99, #0099ff);
        color: white;
        font-size: 1.2rem;
        height: 3.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧪 Анализ руды (MVP)")
st.caption("Интеллектуальный анализ шлифа с цветовой сегментацией")

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

# -----------------------------
# 2. Парсинг изображения (Твой исходный лоадер метаданных)
# -----------------------------
img_data = load_image(file)

# Инициализируем процессор для масок
processor = OreProcessor()
# Считываем байты из uploader'а и передаем в OpenCV пайплайн
file.seek(0)
file_bytes = file.read()
processor.load_from_bytes(file_bytes)

# -----------------------------
# 3. Две колонки (Исходное и ЧБ/Предобработанное через CLAHE)
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Исходное изображение")
    st.image(file, use_container_width=True)

with col2:
    st.subheader("⚙️ Предобработанное изображение (Грейскейл + CLAHE)")
    # Применяем шаг нормализации яркости из твоего второго скрипта перед показом
    processor.apply_clahe_preprocessing(target_brightness=120)
    st.image(processor.gray, use_container_width=True, channels="GRAY")

# -----------------------------
# 4. Метаданные
# -----------------------------
st.subheader("📊 Метаданные изображения")
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
# 5. Реальный компьютерный анализ
# -----------------------------
if st.button("🔬 Запустить анализ", type="primary", use_container_width=True):
    with st.spinner("Выполняется математический анализ масок минералов..."):
        # Рассчитываем маски на основе CLAHE-изображения
        processor.create_talc_mask()
        processor.create_fine_mask()
        processor.create_normal_mask()
        
        # Получаем реальную статистику по пикселям
        stats = processor.get_statistics()
        ore_class = processor.classify_ore(stats)
        
        time.sleep(0.5) # UX-пауза
        
        st.success(f"✅ Анализ завершён! Тип породы: **{ore_class}**")
        
        # Верхний блок с метриками
        st.subheader("📊 Результаты количественного анализа")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("🔵 Тёмные области (Тальк)", f"{stats['talc']}%")
        m_col2.metric("🔴 Тонкие срастания (Fine)", f"{stats['fine']}%")
        m_col3.metric("🟢 Крупные области (Normal)", f"{stats['normal']}%")
        m_col4.metric("⚪ Пустая порода / Остальное", f"{stats['waste']}%")
        
        st.divider()
        
# Интерактивные вкладки для раздельного просмотра масок
        st.subheader("🎨 Визуализация сегментации")
        
        # Создаем 4 вкладки: Общая и 3 раздельные
        tab_all, tab_talc, tab_fine, tab_normal = st.tabs([
            "🌈 Все маски вместе", 
            "🔵 Только Тальк", 
            "🔴 Тонкие срастания", 
            "🟢 Крупные области"
        ])
        
        # Базовые цвета для создания индивидуальных оверлеев
        colors_rgb = {
            "talc": [0, 0, 255],     # Синий
            "fine": [255, 0, 0],     # Красный
            "normal": [0, 255, 0]    # Зеленый
        }
        
        # Задаем желаемую ширину изображения в пикселях (например, 600px или 700px)
        # Это не даст картинке растягиваться на весь экран монитора
        IMG_DISPLAY_WIDTH = 650 
        
        with tab_all:
            annotated_img = processor.get_combined_overlay()
            # Помещаем в колонки, чтобы картинка была компактной
            c_img, _ = st.columns([2, 1])
            with c_img:
                st.image(annotated_img, caption="Совмещенная карта сегментации", width=IMG_DISPLAY_WIDTH)
            
        with tab_talc:
            if processor.masks["talc"] is not None:
                overlay_talc = processor.rgb.copy()
                overlay_talc[processor.masks["talc"] > 0] = colors_rgb["talc"]
                c_img, _ = st.columns([2, 1])
                with c_img:
                    st.image(overlay_talc, caption="Выделен только Тальк (Синий цвет)", width=IMG_DISPLAY_WIDTH)
            else:
                st.warning("Маска Талька пуста")
        
        with tab_fine:
            if processor.masks["fine"] is not None:
                overlay_fine = processor.rgb.copy()
                overlay_fine[processor.masks["fine"] > 0] = colors_rgb["fine"]
                c_img, _ = st.columns([2, 1])
                with c_img:
                    st.image(overlay_fine, caption="Выделены только Тонкие срастания (Красный цвет)", width=IMG_DISPLAY_WIDTH)
            else:
                st.warning("Маска тонких срастаний пуста")
                
        with tab_normal:
            if processor.masks["normal"] is not None:
                overlay_normal = processor.rgb.copy()
                overlay_normal[processor.masks["normal"] > 0] = colors_rgb["normal"]
                c_img, _ = st.columns([2, 1])
                with c_img:
                    st.image(overlay_normal, caption="Выделены только Крупные области (Зеленый цвет)", width=IMG_DISPLAY_WIDTH)
            else:
                st.warning("Маска крупных областей пуста")

        # График распределения
        st.subheader("📈 Распределение компонентов")
        chart_data = {
            "Компонент": ["Тальк", "Тонкие ср.", "Крупные ср.", "Остальное"],
            "Процент (%)": [stats['talc'], stats['fine'], stats['normal'], stats['waste']]
        }
        st.bar_chart(chart_data, x="Компонент", y="Percent" if "Percent" in chart_data else "Процент (%)")

else:
    st.info("Нажмите кнопку «Запустить анализ» для получения реальных результатов")