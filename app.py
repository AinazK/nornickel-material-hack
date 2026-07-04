from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import sys
import os
import io
import time
import base64
import json
import shutil
import stat
import importlib.util
from pathlib import Path
from datetime import datetime

import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from typing import List, Tuple, Optional
import segmentation_models_pytorch as smp

# ==========================================================
# 1. ИНИЦИАЛИЗАЦИЯ ПУТЕЙ И ДИНАМИЧЕСКИЕ ИМПОРТЫ
# ==========================================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
UTILS_PATH = PROJECT_ROOT / "utils"

for p in [SRC_PATH, UTILS_PATH]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transforms import get_train_transforms, get_val_transforms
from imageLoader import load_image

# ==========================================================
# 2. АВТОМАТИЧЕСКАЯ ЗАГРУЗКА МОДЕЛИ (JOBLOAD / TORCH)
# ==========================================================
@st.cache_resource
def load_best_model_automatically():
    """Автоматический поиск и загрузка лучшей модели."""
    import joblib

    checkpoints_dir = PROJECT_ROOT / "artifacts" / "checkpoints"
    model_files = list(checkpoints_dir.glob("best_model_*.pth"))

    if not model_files:
        return None, None, None

    latest_model_path = max(model_files, key=os.path.getmtime)

    try:
        checkpoint = joblib.load(latest_model_path)
    except Exception:
        checkpoint = torch.load(latest_model_path, map_location="cpu")

    config = checkpoint.get("config", {})
    target_class = config.get("target_class", "talc")
    image_size = config.get("image_size", 256)

    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, config, latest_model_path.name


MODEL, MODEL_CONFIG, MODEL_FILENAME = load_best_model_automatically()

if MODEL is not None:
    st.session_state["auto_loaded_model"] = MODEL
    st.session_state["auto_loaded_config"] = MODEL_CONFIG


# ==========================================================
# 3. ИНФЕРЕНС U-NET НА ИЗОБРАЖЕНИИ
# ==========================================================
def predict_unet_on_image(model: torch.nn.Module, image: np.ndarray,
                          device: str = "cpu", image_size: int = 256) -> np.ndarray:
    """
    Предсказание бинарной маски на одном изображении с помощью U-Net.
    Возвращает маску того же размера, что и входное изображение.
    """
    transform = get_val_transforms(image_size=image_size)
    empty_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    augmented = transform(image=image, mask=empty_mask)
    img_tensor = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Ресайз предсказания до исходного размера изображения
    if pred.shape != image.shape[:2]:
        pred = np.array(
            Image.fromarray(pred.astype(np.uint8) * 255).resize(
                (image.shape[1], image.shape[0]), Image.NEAREST
            ),
            dtype=np.uint8
        )
        pred = (pred > 127).astype(np.uint8)

    return pred


# ==========================================================
# 4. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ГЕНЕРАЦИЯ ОТЧЕТОВ
# ==========================================================
def safe_rmtree(path: str, retries: int = 3, delay: float = 0.5):
    if not os.path.exists(path): return

    def _remove_readonly(func, path, _):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    for attempt in range(retries):
        try:
            if sys.version_info >= (3, 12):
                shutil.rmtree(path, onexc=_remove_readonly)
            else:
                shutil.rmtree(path, onerror=_remove_readonly)
            return
        except OSError:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                shutil.rmtree(path, ignore_errors=True)


def generate_report_export(unet_stats, manual_stats, class_iou, class_dice,
                           ore_class, image_filename, target_class):
    """
    Универсальный генератор отчёта с поддержкой кириллицы.
    Всегда генерирует PDF через matplotlib.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib
    matplotlib.use('Agg')  # Неинтерактивный бэкенд
    from matplotlib import rcParams
    from matplotlib.font_manager import FontProperties
    import os

    # ============================================================
    # ПОИСК И РЕГИСТРАЦИЯ ШРИФТА С ПОДДЕРЖКОЙ КИРИЛЛИЦЫ
    # ============================================================
    # Список возможных путей к шрифтам с кириллицей
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",  # Windows - Arial
        "C:/Windows/Fonts/times.ttf",  # Windows - Times New Roman
        "C:/Windows/Fonts/dejavu/DejaVuSans.ttf",  # Windows - DejaVu
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Linux (Arch)
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "/System/Library/Fonts/Arial.ttf",  # macOS
    ]

    # Находим первый доступный шрифт
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break

    # Если шрифт найден, регистрируем его
    if font_path:
        font_prop = FontProperties(fname=font_path)
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [font_prop.get_name()]
        # Принудительно устанавливаем шрифт для всех элементов
        matplotlib.rcParams['axes.unicode_minus'] = False  # Для корректного отображения минуса
    else:
        # Фолбэк на стандартный шрифт (кириллица может не работать)
        font_prop = None

    buf = io.BytesIO()

    with PdfPages(buf) as pdf:
        # ============================================================
        # СТРАНИЦА 1: Титульная и метаданные
        # ============================================================
        fig_title = plt.figure(figsize=(210 / 25.4, 297 / 25.4))  # Формат A4
        fig_title.patch.set_facecolor('#f8f9fa')

        # Используем fontproperties для каждого текста
        fp_title = FontProperties(fname=font_path, size=22, weight='bold') if font_path else None
        fp_subtitle = FontProperties(fname=font_path, size=14) if font_path else None
        fp_normal = FontProperties(fname=font_path, size=12) if font_path else None
        fp_bold = FontProperties(fname=font_path, size=12, weight='bold') if font_path else None

        plt.figtext(0.5, 0.85, "ОТЧЁТ ПО АНАЛИЗУ РУДНОГО ШЛИФА",
                    ha='center', fontproperties=fp_title, color='#1E88E5')
        plt.figtext(0.5, 0.78, "U-Net (ResNet50 Encoder) — семантическая сегментация",
                    ha='center', fontproperties=fp_subtitle, color='#424242')

        plt.figtext(0.15, 0.65, "Дата анализа:", fontproperties=fp_bold)
        plt.figtext(0.35, 0.65, datetime.now().strftime('%d.%m.%Y %H:%M'), fontproperties=fp_normal)

        plt.figtext(0.15, 0.60, "Имя файла:", fontproperties=fp_bold)
        plt.figtext(0.35, 0.60, image_filename, fontproperties=fp_normal)

        plt.figtext(0.15, 0.55, "Целевой класс модели:", fontproperties=fp_bold)
        plt.figtext(0.35, 0.55, target_class.upper(), fontproperties=fp_normal, color='#1E88E5')

        plt.figtext(0.15, 0.50, "Классификация породы:", fontproperties=fp_bold)
        plt.figtext(0.35, 0.50, ore_class, fontproperties=fp_normal, color='#43A047')

        plt.figtext(0.5, 0.25, "Система сегментации и анализа руд",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=10, style='italic') if font_path else None,
                    color='#757575')
        plt.figtext(0.5, 0.22, "Разработано в рамках хакатона",
                    ha='center', fontproperties=FontProperties(fname=font_path, size=10) if font_path else None,
                    color='#757575')

        pdf.savefig(fig_title)
        plt.close(fig_title)

        # ============================================================
        # СТРАНИЦА 2: Таблицы результатов
        # ============================================================
        fig_data = plt.figure(figsize=(210 / 25.4, 297 / 25.4))
        fig_data.patch.set_facecolor('white')

        # Заголовок страницы
        plt.figtext(0.5, 0.95, "РЕЗУЛЬТАТЫ СЕГМЕНТАЦИИ",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=18, weight='bold') if font_path else None,
                    color='#1E88E5')

        # --- Таблица 1: U-Net ---
        plt.figtext(0.5, 0.88, "Результаты U-Net сегментации",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=14, weight='bold') if font_path else None,
                    color='#1E88E5')

        unet_table_data = [
            ['Класс', 'Процент площади (%)'],
            [target_class.upper(), f"{unet_stats.get(target_class, 0):.2f}"],
            ['ФОН', f"{unet_stats.get('background', 0):.2f}"]
        ]
        unet_table = plt.table(
            cellText=unet_table_data[1:],
            colLabels=unet_table_data[0],
            cellLoc='center',
            loc='upper center',
            bbox=[0.25, 0.72, 0.5, 0.12]
        )
        unet_table.auto_set_font_size(False)
        unet_table.set_fontsize(11)
        unet_table.scale(1, 1.5)

        # Стилизация заголовка таблицы U-Net
        for (row, col), cell in unet_table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#1E88E5')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#E3F2FD')

        # --- Таблица 2: Ручная разметка ---
        plt.figtext(0.5, 0.65, "Результаты ручной разметки (эксперт)",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=14, weight='bold') if font_path else None,
                    color='#43A047')

        manual_table_data = [['Класс', 'Процент площади (%)']]
        for k, v in manual_stats.items():
            label = "ФОН" if k == "background" else k.upper()
            manual_table_data.append([label, f"{v:.2f}"])

        manual_table = plt.table(
            cellText=manual_table_data[1:],
            colLabels=manual_table_data[0],
            cellLoc='center',
            loc='upper center',
            bbox=[0.25, 0.49, 0.5, 0.12]
        )
        manual_table.auto_set_font_size(False)
        manual_table.set_fontsize(11)
        manual_table.scale(1, 1.5)

        for (row, col), cell in manual_table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#43A047')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#E8F5E9')

        # --- Таблица 3: Метрики ---
        plt.figtext(0.5, 0.42, "Метрики качества совпадения (IoU / Dice)",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=14, weight='bold') if font_path else None,
                    color='#FB8C00')

        metrics_table_data = [['Класс', 'IoU (%)', 'Dice (%)']]
        for k in [target_class, 'background']:
            label = "ФОН" if k == "background" else k.upper()
            metrics_table_data.append([
                label,
                f"{class_iou.get(k, 0):.1f}",
                f"{class_dice.get(k, 0):.1f}"
            ])

        metrics_table = plt.table(
            cellText=metrics_table_data[1:],
            colLabels=metrics_table_data[0],
            cellLoc='center',
            loc='upper center',
            bbox=[0.2, 0.28, 0.6, 0.10]
        )
        metrics_table.auto_set_font_size(False)
        metrics_table.set_fontsize(11)
        metrics_table.scale(1, 1.5)

        for (row, col), cell in metrics_table.get_celld().items():
            if row == 0:
                cell.set_facecolor('#FB8C00')
                cell.set_text_props(color='white', weight='bold')
            else:
                cell.set_facecolor('#FFF3E0')

        # --- Итоговый Dice Index ---
        mean_dice = sum(class_dice.values()) / len(class_dice) if class_dice else 0
        plt.figtext(0.5, 0.18, "Интегральный индекс сходства контуров (Dice Index):",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=13, weight='bold') if font_path else None)
        plt.figtext(0.5, 0.13, f"{mean_dice:.1f}%",
                    ha='center',
                    fontproperties=FontProperties(fname=font_path, size=20, weight='bold') if font_path else None,
                    color='#1E88E5')

        pdf.savefig(fig_data)
        plt.close(fig_data)

    buf.seek(0)
    return buf.getvalue(), "pdf"


# ==========================================================
# 5. ДАТАСЕТ ДЛЯ ОБУЧЕНИЯ
# ==========================================================
class OreDatasetNew(Dataset):
    def __init__(self, data_root: str, target_class: str = "talc", transforms=None, image_size: int = 256):
        self.data_root = Path(data_root)
        self.target_class = target_class
        self.transforms = transforms
        self.image_size = image_size
        self.image_paths: List[Path] = []
        self.mask_paths: List[Path] = []
        self._scan_directory()

    def _scan_directory(self):
        result_folders = [d for d in self.data_root.iterdir() if d.is_dir() and d.name.startswith("output_results")]
        for result_folder in sorted(result_folders):
            numbered_folders = [d for d in result_folder.iterdir() if d.is_dir() and d.name.isdigit()]
            for num_folder in sorted(numbered_folders, key=lambda x: int(x.name)):
                orig_img = None
                for ext in [".JPG", ".jpg", ".jpeg", ".tiff", ".TIF", ".TIFF"]:
                    for img in num_folder.glob(f"*{ext}"):
                        if not img.name.endswith('.png'): orig_img = img; break
                    if orig_img: break
                if not orig_img: continue

                mask_path = num_folder / f"{self.target_class}.png"
                if not mask_path.exists():
                    for m in num_folder.glob("*.png"):
                        if self.target_class.lower() in m.name.lower(): mask_path = m; break
                if mask_path.exists():
                    self.image_paths.append(orig_img)
                    self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        img_path, mask_path = self.image_paths[idx], self.mask_paths[idx]
        img = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask > 0).astype(np.uint8)
        if mask.shape != img.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize((img.shape[1], img.shape[0]), Image.NEAREST), dtype=np.uint8)
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img_tensor, mask_tensor = augmented["image"], augmented["mask"]
        else:
            img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask).long()
        mask_tensor = torch.clamp(mask_tensor.long(), 0, 1)
        return img_tensor, mask_tensor, img_path.name


# ==========================================================
# 6. ПАТЧ ДЛЯ STREAMLIT-DRAWABLE-CANVAS
# ==========================================================
import streamlit.elements.image as _st_image_module

if not hasattr(_st_image_module, "image_to_url"):
    def _image_to_url(image, width=None, clamp=False, channels="RGB", output_format="auto", image_id=None,
                      allow_emoji=False):
        if isinstance(image, Image.Image):
            pil_img = image
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image)
        else:
            pil_img = Image.open(image)
        if pil_img.mode not in ("RGB", "RGBA"): pil_img = pil_img.convert("RGB")
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        data = buffer.getvalue()
        try:
            import streamlit.runtime as _st_runtime
            _media_mgr = _st_runtime.get_instance().media_file_mgr
            return _media_mgr.add(data, "image/png", str(image_id) if image_id else "drawable-canvas-bg")
        except Exception:
            encoded = base64.b64encode(data).decode()
            return f"data:image/png;base64,{encoded}"


    _st_image_module.image_to_url = _image_to_url
from streamlit_drawable_canvas import st_canvas

# ==========================================================
# 7. КОНФИГУРАЦИЯ STREAMLIT И UI
# ==========================================================
st.set_page_config(page_title="Talc Segmentation & Analysis System", page_icon="", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.stButton>button { background: linear-gradient(90deg, #00cc99, #0099ff); color: white; font-size: 1.1rem; height: 3rem; }
.main-header { font-size: 2.5rem; font-weight: 700; color: #1E88E5; margin-bottom: 1rem; text-align: center; }
.sub-header { font-size: 1.2rem; color: #757575; text-align: center; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header"> Система сегментации и анализа руд</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">U-Net (ResNet50) + Экспертный анализ шлифов</p>', unsafe_allow_html=True)
st.markdown("---")

# Индикатор автоматической загрузки модели
if MODEL is not None:
    st.sidebar.success(f"✅ Модель автозагружена: **{MODEL_FILENAME}**")
    st.sidebar.info(f"🎯 Класс: **{MODEL_CONFIG.get('target_class')}** | Размер: {MODEL_CONFIG.get('image_size')}px")
else:
    st.sidebar.warning("⚠️ Файл best_model.pth не найден. Обучите модель или поместите веса в artifacts/checkpoints/")

with st.sidebar:
    st.title("⚙️ Настройки")
    st.subheader("Параметры обучения")
    num_epochs = st.slider("Количество эпох", min_value=1, max_value=100, value=10)
    batch_size = st.selectbox("Размер батча", [1, 2, 4, 8, 16], index=1)
    learning_rate = st.number_input("Learning rate", value=1e-3, format="%.5f")
    image_size = st.selectbox("Размер изображения", [128, 256, 512], index=1)
    target_class = st.selectbox("Целевой класс для сегментации", ["talc", "fine", "normal"], index=0)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    (artifacts_dir / "checkpoints").mkdir(exist_ok=True)
    (artifacts_dir / "results").mkdir(exist_ok=True)

tab1, tab2, tab3 = st.tabs(["📊 Статистика данных", "🚀 Обучение модели", "🧠 Анализ шлифа (U-Net)"])

# ==========================================================
# ВКЛАДКА 1: СТАТИСТИКА ДАННЫХ
# ==========================================================
with tab1:
    st.header("📊 Обзор доступных данных")
    data_dir = PROJECT_ROOT / "data" / "train"

    if not data_dir.exists():
        st.warning("⚠️ Папка data/train не найдена")
        st.info(
            "💡 Эта вкладка требует наличия обучающей выборки. Для демонстрации перейдите во вкладку 'Анализ шлифа (U-Net)'")
    else:
        result_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("output_results")]
        if not result_folders:
            st.warning("⚠️ Папки output_results* не найдены")
            st.info("💡 Для работы необходимы данные в формате output_results*/N/{images + masks}")
        else:
            st.success(f"✅ Найдено {len(result_folders)} наборов данных")
            total_images = 0
            total_stats = {"talc": 0.0, "fine": 0.0, "normal": 0.0}
            classifications = {}

            for result_folder in sorted(result_folders):
                with st.expander(f"📂 {result_folder.name}", expanded=False):
                    json_file = next(result_folder.glob("*.json"), None)
                    if json_file:
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            st.write(f"**Записей в JSON:** {len(data)}")
                            total_images += len(data)
                            for item in data:
                                for mineral, percent in item.get("percentages", {}).items():
                                    if mineral in total_stats:
                                        total_stats[mineral] += percent
                                cls = item.get("classification", "unknown")
                                classifications[cls] = classifications.get(cls, 0) + 1
                            if data:
                                st.write("**Примеры данных:**")
                                st.dataframe(pd.DataFrame(data[:5]), use_container_width=True)
                        except Exception as e:
                            st.error(f"Ошибка чтения JSON: {e}")

            st.divider()
            st.subheader("📈 Общая статистика датасета")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="📁 Всего изображений", value=total_images)
            with col2:
                st.metric(label="🔵 Средний % талька",
                          value=f"{(total_stats['talc'] / total_images if total_images > 0 else 0):.2f}%")
            with col3:
                st.metric(label="🔴 Средний % fine",
                          value=f"{(total_stats['fine'] / total_images if total_images > 0 else 0):.2f}%")
            with col4:
                st.metric(label="🟢 Средний % normal",
                          value=f"{(total_stats['normal'] / total_images if total_images > 0 else 0):.2f}%")
            st.divider()
            st.subheader("📋 Распределение по типам руды")
            if classifications:
                class_df = pd.DataFrame(
                    {'Тип руды': list(classifications.keys()), 'Количество': list(classifications.values())})
                st.dataframe(class_df, use_container_width=True, hide_index=True)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(classifications.values(), labels=classifications.keys(), autopct='%1.1f%%')
                ax.set_title('Распределение образцов по классам')
                st.pyplot(fig)

# ==========================================================
# ВКЛАДКА 2: ОБУЧЕНИЕ МОДЕЛИ
# ==========================================================
with tab2:
    st.header("Обучение модели U-Net (ResNet50 Encoder)")
    data_dir = PROJECT_ROOT / "data" / "train"

    if not data_dir.exists():
        st.warning("⚠️ Папка data/train не найдена")
        st.info(
            "💡 Для обучения модели необходима папка с данными. Перейдите во вкладку 'Анализ шлифа' для использования предобученной модели.")
    else:
        result_folders = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("output_results")]
        if not result_folders:
            st.warning("⚠️ Данные не найдены")
            st.info("💡 Необходимы данные в формате output_results*/N/{images + masks + results.json}")
        else:
            st.info(f"📁 Будут использованы: {', '.join([f.name for f in result_folders])}")
            st.write(f"🎯 Целевой класс: **{target_class}**")

            if st.button("🚀 Начать обучение", type="primary"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                st.info(f"💻 Устройство: {device}")
                with st.spinner("Загрузка датасета..."):
                    try:
                        full_dataset = OreDatasetNew(data_root=str(data_dir), target_class=target_class,
                                                     transforms=get_train_transforms(image_size), image_size=image_size)
                        if len(full_dataset) == 0:
                            st.error("❌ Датасет пуст!");
                            st.stop()
                        train_size = int(0.8 * len(full_dataset));
                        val_size = len(full_dataset) - train_size
                        train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size],
                                                                         generator=torch.Generator().manual_seed(42))
                        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
                        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
                        st.success(f"✅ Датасет загружен: {len(train_ds)} train, {len(val_ds)} val")
                    except Exception as e:
                        st.error(f"❌ Ошибка: {str(e)}");
                        st.stop()
                with st.spinner("Инициализация модели..."):
                    model = smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=2).to(
                        device)
                    criterion = nn.CrossEntropyLoss();
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    st.success("✅ Модель готова")
                st.subheader("Процесс обучения")
                progress_bar = st.progress(0);
                status_text = st.empty();
                chart_placeholder = st.empty()
                best_val_loss = float("inf");
                best_epoch = 0;
                chart_data = []
                for epoch in range(num_epochs):
                    model.train();
                    epoch_loss = 0.0
                    for images, masks, _ in train_loader:
                        images, masks = images.to(device), masks.long().to(device)
                        optimizer.zero_grad();
                        loss = criterion(model(images), masks);
                        loss.backward();
                        optimizer.step()
                        epoch_loss += loss.item()
                    model.eval();
                    val_loss = 0.0
                    with torch.no_grad():
                        for images, masks, _ in val_loader:
                            images, masks = images.to(device), masks.long().to(device)
                            val_loss += criterion(model(images), masks).item()
                    avg_train, avg_val = epoch_loss / len(train_loader), val_loss / len(val_loader)
                    progress_bar.progress((epoch + 1) / num_epochs)
                    status_text.text(f"Epoch {epoch + 1}/{num_epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
                    chart_data.append({"epoch": epoch + 1, "train_loss": avg_train, "val_loss": avg_val})
                    if avg_val < best_val_loss:
                        best_val_loss = avg_val;
                        best_epoch = epoch + 1
                        checkpoint_path = artifacts_dir / "checkpoints" / f"best_model_{target_class}.pth"
                        torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(), "val_loss": best_val_loss,
                                    "config": {"num_classes": 2, "image_size": image_size, "encoder": "resnet50",
                                               "target_class": target_class}}, checkpoint_path)
                    chart_placeholder.line_chart(pd.DataFrame(chart_data).set_index("epoch"))
                st.success(f"✅ Обучение завершено! Лучшая эпоха: {best_epoch} (Val Loss: {best_val_loss:.4f})")
                checkpoint_path = artifacts_dir / "checkpoints" / f"best_model_{target_class}.pth"
                if checkpoint_path.exists():
                    with open(checkpoint_path, "rb") as f:
                        st.download_button(label="⬇️ Скачать best_model.pth", data=f,
                                           file_name=f"best_model_{target_class}.pth", mime="application/octet-stream")

# ==========================================================
# ВКЛАДКА 3: АНАЛИЗ ШЛИФА (U-NET + РУЧНАЯ РАЗМЕТКА)
# ==========================================================
with tab3:
    st.header("🧠 Анализ шлифа: U-Net vs Ручная разметка")

    # Проверка наличия модели
    if MODEL is None:
        st.error(
            "❌ Модель U-Net не загружена. Сначала обучите модель во вкладке 'Обучение модели' или поместите файл best_model.pth в artifacts/checkpoints/")
        st.stop()

    # Инфо о модели
    target_class_model = MODEL_CONFIG.get('target_class', 'talc')
    image_size_model = MODEL_CONFIG.get('image_size', 256)
    st.info(f"🎯 Активная модель обучена на классе: **{target_class_model}** (размер входа: {image_size_model}px)")

    file = st.file_uploader("Выберите изображение шлифа", type=["png", "jpg", "jpeg", "tiff"])
    if file is None: st.info("Загрузите изображение для начала анализа"); st.stop()

    if st.session_state.get("_current_file") != file.name:
        st.session_state["_current_file"] = file.name
        st.session_state["_canvas_key_suffix"] = 0
        st.session_state.pop("manual_stats", None)
        st.session_state.pop("unet_stats", None)
        st.session_state.pop("unet_mask", None)

    img_data = load_image(file)
    file.seek(0);
    file_bytes = file.read()
    original_pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    original_np = np.array(original_pil)

    # Цвета для отображения
    CLASS_COLORS = {
        "talc": [0, 0, 255],  # Синий
        "fine": [255, 0, 0],  # Красный
        "normal": [0, 255, 0],  # Зелёный
    }
    CLASS_CSS = {
        "talc": "rgb(0, 0, 255)",
        "fine": "rgb(255, 0, 0)",
        "normal": "rgb(0, 255, 0)",
    }
    CLASS_LABELS = {"talc": "🔵 Тальк", "fine": "🔴 Тонкие срастания", "normal": "🟢 Крупные области"}

    target_color = CLASS_COLORS.get(target_class_model, [255, 255, 0])
    target_css = CLASS_CSS.get(target_class_model, "rgb(255, 255, 0)")
    target_label = CLASS_LABELS.get(target_class_model, target_class_model)

    # ==========================================================
    # 1. ИНФЕРЕНС U-NET
    # ==========================================================
    st.subheader(f"1️⃣ Сегментация U-Net (целевой класс: {target_label})")
    if st.button("🧠 Запустить U-Net сегментацию", type="primary", use_container_width=True):
        with st.spinner("Выполняется инференс нейросети..."):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = MODEL.to(device)
            model.eval()

            unet_mask = predict_unet_on_image(model, original_np, device, image_size_model)

            # Вычисление статистики
            total_pixels = unet_mask.shape[0] * unet_mask.shape[1]
            target_pixels = int(np.count_nonzero(unet_mask))
            background_pixels = total_pixels - target_pixels

            unet_stats = {
                target_class_model: round((target_pixels / total_pixels) * 100, 2),
                "background": round((background_pixels / total_pixels) * 100, 2)
            }

            # Классификация руды (упрощённая, на основе процентов)
            if target_class_model == "talc":
                ore_class = "оталькованная руда (talc)" if unet_stats["talc"] > 10 else "рядовая руда (normal)"
            elif target_class_model == "fine":
                ore_class = "труднообогатимая руда (fine)" if unet_stats["fine"] > 30 else "рядовая руда (normal)"
            else:
                ore_class = "рядовая руда (normal)"

            st.session_state["unet_stats"] = unet_stats
            st.session_state["unet_mask"] = unet_mask
            st.session_state["ore_class"] = ore_class

    if "unet_stats" in st.session_state:
        unet_stats = st.session_state["unet_stats"]
        unet_mask = st.session_state["unet_mask"]
        ore_class = st.session_state["ore_class"]

        st.success(f"✅ Сегментация завершена! Тип породы: **{ore_class}**")

        # Метрики
        m_col1, m_col2 = st.columns(2)
        m_col1.metric(label=target_label, value=f"{unet_stats[target_class_model]:.2f}%")
        m_col2.metric(label=" Фон / Остальное", value=f"{unet_stats['background']:.2f}%")

        # Визуализация
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал")
            st.image(original_pil, use_container_width=True)
        with col2:
            st.subheader(f"Маска U-Net ({target_label})")
            st.image(unet_mask, caption="Белый = целевой класс", use_container_width=True)

        # Наложение
        overlay = original_np.copy()
        overlay[unet_mask > 0] = target_color
        st.subheader("Наложение маски на оригинал")
        st.image(overlay, caption=f"{target_label} выделен цветом", use_container_width=True)

    st.divider()

    # ==========================================================
    # 2. РУЧНАЯ РАЗМЕТКА
    # ==========================================================
    st.subheader("2️⃣ Ручная разметка (Эксперт)")
    DISPLAY_WIDTH = 600
    w, h = original_pil.size
    scale = DISPLAY_WIDTH / w
    canvas_height = int(h * scale)
    pil_img_resized = original_pil.resize((DISPLAY_WIDTH, canvas_height))

    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        active_label = st.selectbox(
            "Что вы хотите разметить сейчас?",
            options=[CLASS_LABELS["talc"], CLASS_LABELS["fine"], CLASS_LABELS["normal"]],
            key="active_class_selector"
        )
        current_class = "talc" if active_label == CLASS_LABELS["talc"] else (
            "fine" if active_label == CLASS_LABELS["fine"] else "normal")
    with ctrl_col2:
        brush_size = st.slider("Размер кисти", 3, 40, 12)

    if st.button("🗑️ Очистить разметку"):
        st.session_state["_canvas_key_suffix"] = st.session_state.get("_canvas_key_suffix", 0) + 1
        st.session_state.pop("manual_stats", None)

    st.info(f" Сейчас активна кисть: {active_label}")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=brush_size,
        stroke_color=CLASS_CSS[current_class],
        background_image=pil_img_resized,
        update_streamlit=True,
        height=canvas_height,
        width=DISPLAY_WIDTH,
        drawing_mode="freedraw",
        key=f"single_canvas_{st.session_state.get('_current_file')}_{st.session_state.get('_canvas_key_suffix', 0)}"
    )


    def compute_manual_stats_single(canvas_data):
        if canvas_data is None or canvas_data.image_data is None: return None
        img = canvas_data.image_data
        total_pixels = img.shape[0] * img.shape[1]
        alpha = img[:, :, 3] > 0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        stats = {
            "talc": round(100 * (alpha & (b > 200) & (r < 50) & (g < 50)).sum() / total_pixels, 2),
            "fine": round(100 * (alpha & (r > 200) & (g < 50) & (b < 50)).sum() / total_pixels, 2),
            "normal": round(100 * (alpha & (g > 200) & (r < 50) & (b < 50)).sum() / total_pixels, 2),
        }
        stats["background"] = round(max(0.0, 100 - sum(stats.values())), 2)
        return stats


    if st.button("✅ Зафиксировать ручную разметку", use_container_width=True):
        result = compute_manual_stats_single(canvas_result)
        if result and (result["talc"] > 0 or result["fine"] > 0 or result["normal"] > 0):
            st.session_state["manual_stats"] = result
        else:
            st.warning("Сначала закрасьте хотя бы одну область!")

    if "manual_stats" in st.session_state:
        mstats = st.session_state["manual_stats"]
        st.subheader("Статистика ручной разметки")
        mm_col1, mm_col2, mm_col3, mm_col4 = st.columns(4)
        mm_col1.metric(CLASS_LABELS["talc"], f"{mstats['talc']}%")
        mm_col2.metric(CLASS_LABELS["fine"], f"{mstats['fine']}%")
        mm_col3.metric(CLASS_LABELS["normal"], f"{mstats['normal']}%")
        mm_col4.metric("⚪ Фон", f"{mstats['background']}%")

    st.divider()

    # ==========================================================
    # 3. СРАВНЕНИЕ U-NET И РУЧНОЙ РАЗМЕТКИ
    # ==========================================================
    st.subheader("3️⃣ Сравнение U-Net и ручной разметки")
    if "unet_stats" in st.session_state and "manual_stats" in st.session_state:
        unet_stats = st.session_state["unet_stats"]
        manual_stats = st.session_state["manual_stats"]
        unet_mask = st.session_state["unet_mask"]
        img_canvas = canvas_result.image_data if (
                    canvas_result is not None and canvas_result.image_data is not None) else None

        class_iou = {}
        class_dice = {}

        if img_canvas is not None:
            canvas_h, canvas_w = img_canvas.shape[0], img_canvas.shape[1]
            alpha = img_canvas[:, :, 3] > 0
            r_c, g_c, b_c = img_canvas[:, :, 0], img_canvas[:, :, 1], img_canvas[:, :, 2]

            # Маски ручной разметки (по всем трём классам)
            manual_masks_bin = {
                "talc": alpha & (b_c > 200) & (r_c < 50) & (g_c < 50),
                "fine": alpha & (r_c > 200) & (g_c < 50) & (b_c < 50),
                "normal": alpha & (g_c > 200) & (r_c < 50) & (b_c < 50),
            }
            manual_masks_bin["background"] = ~alpha

            # Маска U-Net (бинарная: целевой класс vs фон)
            unet_mask_resized = cv2.resize(unet_mask, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
            unet_mask_bin = unet_mask_resized > 0

            algo_masks_bin = {
                target_class_model: unet_mask_bin,
                "background": ~unet_mask_bin
            }
            # Добавляем нулевые маски для классов, которые не предсказывает модель
            for cls in ["talc", "fine", "normal"]:
                if cls != target_class_model:
                    algo_masks_bin[cls] = np.zeros((canvas_h, canvas_w), dtype=bool)

            # Вычисление метрик для целевого класса и фона
            for key in [target_class_model, "background"]:
                m_mask = manual_masks_bin[key]
                a_mask = algo_masks_bin[key]
                intersection = np.logical_and(m_mask, a_mask).sum()
                union = np.logical_or(m_mask, a_mask).sum()
                iou = 100.0 if union == 0 else (intersection / union) * 100
                total_elements = m_mask.sum() + a_mask.sum()
                dice = 100.0 if total_elements == 0 else ((2.0 * intersection) / total_elements) * 100
                class_iou[key] = round(iou, 1)
                class_dice[key] = round(dice, 1)

        # Таблица сравнения
        rows = []
        for key in [target_class_model, "background"]:
            rows.append({
                "Класс": target_label if key == target_class_model else "⚪ Фон",
                "U-Net (%)": float(unet_stats.get(key, 0)),
                "Ручная разметка (%)": float(manual_stats.get(key, 0)),
                "Разница (п.п.)": round(float(unet_stats.get(key, 0)) - float(manual_stats.get(key, 0)), 2),
                "IoU (%)": class_iou.get(key, 0.0),
                "Dice (%)": class_dice.get(key, 0.0)
            })

        st.data_editor(rows, column_config={
            "Разница (п.п.)": st.column_config.NumberColumn(format="%+.2f"),
            "IoU (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
            "Dice (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")
        }, use_container_width=True, hide_index=True, disabled=True)

        # Итоговый Dice Index
        mean_dice = round(sum(class_dice.values()) / len(class_dice), 1) if class_dice else 0
        if mean_dice >= 80:
            bar_color, status_text, alert_type = "#00cc99", "Отличная точность!", st.success
        elif mean_dice >= 50:
            bar_color, status_text, alert_type = "#ffcc00", "Умеренное пространственное сходство.", st.warning
        else:
            bar_color, status_text, alert_type = "#ff3333", "Низкое совпадение контуров.", st.error

        st.markdown(
            f"""<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 8px solid {bar_color};"><div style="display: flex; justify-content: space-between; align-items: center;"><span style="font-size: 1.2rem; font-weight: bold;">Интегральный индекс сходства контуров (Dice Index):</span><span style="font-size: 2rem; font-weight: bold; color: {bar_color};">{mean_dice}%</span></div><div style="background-color: #e0e0e0; border-radius: 5px; height: 15px; width: 100%; margin-top: 10px; overflow: hidden;"><div style="background-color: {bar_color}; width: {mean_dice}%; height: 100%; border-radius: 5px; transition: width 0.5s ease-in-out;"></div></div></div>""",
            unsafe_allow_html=True)
        alert_type(status_text)

        # ЭКСПОРТ ОТЧЕТА
        st.divider()
        st.subheader("📄 Экспорт результатов")
        report_data, report_format = generate_report_export(
            unet_stats=unet_stats, manual_stats=manual_stats,
            class_iou=class_iou, class_dice=class_dice,
            ore_class=st.session_state.get("ore_class", "Не определено"),
            image_filename=file.name, target_class=target_class_model
        )

        if report_format == "pdf":
            st.download_button(label="📥 Скачать PDF отчет", data=report_data,
                               file_name=f"unet_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf", use_container_width=True, type="primary")
        else:
            st.download_button(label="📥 Скачать PNG отчет", data=report_data,
                               file_name=f"unet_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                               mime="image/png", use_container_width=True, type="primary")
    else:
        st.info("Чтобы увидеть сравнение, выполните U-Net сегментацию и ручную разметку.")

# ==========================================================
# ПОДВАЛ
# ==========================================================
st.markdown("---")
st.markdown(
    """<div style='text-align: center; color: gray;'><p>🔬 Система сегментации и анализа руд | U-Net с ResNet50 encoder</p><p>Разработано для хакатона</p></div>""",
    unsafe_allow_html=True)