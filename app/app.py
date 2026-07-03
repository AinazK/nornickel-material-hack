import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app'))

import streamlit as st
from utils.ainaz.imageLoader import load_image

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

# Заголовок точно как на картинке
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
# 2. Парсинг изображения
# -----------------------------
img_data = load_image(file)

# -----------------------------
# 3. Две колонки
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Исходное изображение")
    st.image(file, use_container_width=True)

with col2:
    st.subheader("⚙️ Обработанное изображение")
    st.image(img_data.image, use_container_width=True)

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
# 5. Анализ
# -----------------------------
if st.button("🔬 Запустить анализ", type="primary", use_container_width=True):
    with st.spinner("Выполняется анализ..."):
        time.sleep(1.5)
        
        st.success("✅ Анализ завершён!")
        
        st.subheader("🎨 Результат сегментации")
        
        res_col1, res_col2 = st.columns([3, 2])
        
        with res_col1:
            st.image(
                "https://via.placeholder.com/900x480/008855/ffffff?text=Цветная+Маска+Сегментации",
                caption="Цветная маска сегментации",
                use_container_width=True
            )
        
        with res_col2:
            st.metric("🟢 Пирит", "42.8%")
            st.metric("🔵 Халькопирит", "31.5%")
            st.metric("🔴 Сфалерит", "18.7%")
            st.metric("⚪ Пустая порода", "7.0%")
            
            st.divider()
            st.success("**Точность модели:** 94.2%")

        st.subheader("📈 Распределение минералов")
        chart_data = {
            "Минерал": ["Пирит", "Халькопирит", "Сфалерит", "Пустая порода"],
            "Процент": [42.8, 31.5, 18.7, 7.0]
        }
        st.bar_chart(chart_data, x="Минерал", y="Процент")

else:
    st.info("Нажмите кнопку «Запустить анализ» для получения результата")