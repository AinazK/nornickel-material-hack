import streamlit as st
from utils.ainaz.imageLoader import load_image

st.set_page_config(
    page_title="Nornickel Material Hack",
    layout="wide"
)

st.title("🧪 Анализ руды (MVP)")
st.caption("Загрузка и первичный разбор изображения шлифа")

# -----------------------------
# 1. Загрузка файла
# -----------------------------
file = st.file_uploader(
    "Выберите изображение",
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
# 3. Две колонки: вход + обработка
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📥 Исходное изображение")
    st.image(file, use_container_width=True)

with col2:
    st.subheader("⚙️ Numpy представление")
    st.image(img_data.image, use_container_width=True)

# -----------------------------
# 4. Полная информация (ImageData)
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

# -----------------------------
# 5. Технический debug-блок
# -----------------------------
with st.expander("🔍 Debug информация (для команды ML)"):
    st.write("Shape:", img_data.image.shape)
    st.write("dtype:", img_data.image.dtype)
    st.write("min pixel:", img_data.image.min())
    st.write("max pixel:", img_data.image.max())

# -----------------------------
# 6. ML кнопка (заглушка)
# -----------------------------
st.divider()

if st.button("🔬 Запустить анализ"):
    st.info("Пайплайн: загрузка → preprocess → segmentation (будет подключено SegFormer)")