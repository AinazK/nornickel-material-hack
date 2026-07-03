#!/usr/bin/env python3
"""
Скрипт для оценки точности выделения маски.
Учитывает три типа разметки на GT-изображении:
  - Зелёный  : обычные срастания
  - Красный  : тонкие срастания
  - Синий    : тальк

Метрики:
  Сегментация : IoU, Hausdorff Distance (для каждого класса и средние)
  Классификация: F1-score, AUC-ROC (для каждого класса и средние)

Зависимости:
pip install numpy opencv-python scikit-image scipy scikit-learn matplotlib
"""
import numpy as np
import cv2
from scipy import ndimage
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # неблокирующий бэкенд — важно для серверов
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.table import Table
from typing import Tuple, Dict, Optional, List
import os


# ============================================================
# 1. КОНСТАНТЫ ЦВЕТОВЫХ ДИАПАЗОНОВ (BGR)
# ============================================================
COLOR_RANGES = {
    "green": {
        "lower": (30, 80, 80),
        "upper": (120, 255, 255),
        "label": "Обычные срастания",
        "bgr": (0, 255, 0),
        "mpl": "#00FF00",   # для matplotlib
    },
    "red": {
        "lower": (30, 30, 150),
        "upper": (120, 120, 255),
        "label": "Тонкие срастания",
        "bgr": (0, 0, 255),
        "mpl": "#FF0000",
    },
    "blue": {
        "lower": (150, 50, 50),
        "upper": (255, 150, 120),
        "label": "Тальк",
        "bgr": (255, 0, 0),
        "mpl": "#0000FF",
    },
}

CLASS_NAMES = ["green", "red", "blue"]


# ============================================================
# 2. ЗАГРУЗКА И ПОДГОТОВКА ИЗОБРАЖЕНИЙ
# ============================================================
def load_images(path_gt: str, path_pred: str) -> Tuple[np.ndarray, np.ndarray]:
    img_gt = cv2.imread(path_gt, cv2.IMREAD_COLOR)
    img_pred = cv2.imread(path_pred, cv2.IMREAD_COLOR)
    if img_gt is None:
        raise FileNotFoundError(f"Не удалось загрузить GT-изображение: {path_gt}")
    if img_pred is None:
        raise FileNotFoundError(f"Не удалось загрузить предсказание: {path_pred}")
    h, w = img_gt.shape[:2]
    img_pred = cv2.resize(img_pred, (w, h), interpolation=cv2.INTER_AREA)
    return img_gt, img_pred


# ============================================================
# 3. ИЗВЛЕЧЕНИЕ МАСОК ПО ТРЁМ ЦВЕТАМ
# ============================================================
def extract_color_masks(img: np.ndarray,
                        dilate_px: int = 3,
                        fill_holes: bool = True) -> Dict[str, np.ndarray]:
    masks = {}
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    for name, spec in COLOR_RANGES.items():
        lower = np.array(spec["lower"], dtype=np.uint8)
        upper = np.array(spec["upper"], dtype=np.uint8)
        mask_color = cv2.inRange(img, lower, upper)
        mask_dilated = cv2.dilate(mask_color, kernel, iterations=2)
        if fill_holes:
            contours, _ = cv2.findContours(
                mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            filled = np.zeros_like(mask_dilated)
            cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
            masks[name] = filled
        else:
            masks[name] = mask_dilated
    return masks


def extract_combined_mask(masks: Dict[str, np.ndarray]) -> np.ndarray:
    combined = np.zeros_like(next(iter(masks.values())))
    for m in masks.values():
        combined = cv2.bitwise_or(combined, m)
    return combined


# ============================================================
# 4. МЕТРИКИ СЕГМЕНТАЦИИ
# ============================================================
def compute_iou(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    gt_bin = (mask_gt > 0).astype(np.uint8)
    pr_bin = (mask_pred > 0).astype(np.uint8)
    intersection = np.logical_and(gt_bin, pr_bin).sum()
    union = np.logical_or(gt_bin, pr_bin).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def compute_hausdorff(mask_gt: np.ndarray,
                      mask_pred: np.ndarray,
                      percentile: int = 95) -> float:
    gt_bin = (mask_gt > 0).astype(np.uint8)
    pr_bin = (mask_pred > 0).astype(np.uint8)
    gt_edges = cv2.Canny(gt_bin * 255, 50, 150)
    pr_edges = cv2.Canny(pr_bin * 255, 50, 150)
    gt_pts = np.argwhere(gt_edges > 0)
    pr_pts = np.argwhere(pr_edges > 0)
    if len(gt_pts) == 0 or len(pr_pts) == 0:
        return float('inf')
    tree_gt = cKDTree(gt_pts)
    tree_pr = cKDTree(pr_pts)
    dist_pr_to_gt, _ = tree_gt.query(pr_pts)
    dist_gt_to_pr, _ = tree_pr.query(gt_pts)
    all_dists = np.concatenate([dist_pr_to_gt, dist_gt_to_pr])
    return float(np.percentile(all_dists, percentile))


# ============================================================
# 5. МЕТРИКИ КЛАССИФИКАЦИИ
# ============================================================
def compute_f1(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    gt_flat = (mask_gt > 0).astype(np.uint8).flatten()
    pr_flat = (mask_pred > 0).astype(np.uint8).flatten()
    return float(f1_score(gt_flat, pr_flat, zero_division=0))


def compute_auc(mask_gt: np.ndarray, mask_pred: np.ndarray) -> float:
    gt_flat = (mask_gt > 0).astype(np.uint8).flatten()
    pr_bin = (mask_pred > 0).astype(np.uint8)
    dist_fg = ndimage.distance_transform_edt(pr_bin)
    dist_bg = ndimage.distance_transform_edt(1 - pr_bin)
    confidence = dist_fg.astype(np.float32)
    confidence[pr_bin == 0] = -dist_bg[pr_bin == 0]
    conf_flat = confidence.flatten()
    cmin, cmax = conf_flat.min(), conf_flat.max()
    if cmax - cmin > 0:
        conf_norm = (conf_flat - cmin) / (cmax - cmin)
    else:
        conf_norm = conf_flat
    try:
        return float(roc_auc_score(gt_flat, conf_norm))
    except ValueError:
        return 0.0


def compute_confusion(mask_gt: np.ndarray, mask_pred: np.ndarray) -> Dict[str, int]:
    gt_flat = (mask_gt > 0).astype(np.uint8).flatten()
    pr_flat = (mask_pred > 0).astype(np.uint8).flatten()
    tn, fp, fn, tp = confusion_matrix(gt_flat, pr_flat, labels=[0, 1]).ravel()
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)}


def compute_metrics_for_class(mask_gt: np.ndarray,
                              mask_pred: np.ndarray,
                              hausdorff_percentile: int = 95) -> Dict[str, float]:
    cm = compute_confusion(mask_gt, mask_pred)
    precision = cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0.0
    recall = cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0.0
    return {
        "IoU": compute_iou(mask_gt, mask_pred),
        "Hausdorff_px": compute_hausdorff(mask_gt, mask_pred, hausdorff_percentile),
        "F1": compute_f1(mask_gt, mask_pred),
        "AUC": compute_auc(mask_gt, mask_pred),
        "Precision": precision,
        "Recall": recall,
        "TP": cm["TP"], "FP": cm["FP"],
        "TN": cm["TN"], "FN": cm["FN"],
    }


# ============================================================
# 6. ПОЛНЫЙ ОТЧЁТ
# ============================================================
def evaluate_segmentation(path_gt_img: str,
                          path_pred_img: str,
                          hausdorff_percentile: int = 95,
                          show_visualization: bool = True,
                          save_viz_path: Optional[str] = "segmentation_viz.png",
                          save_report: Optional[str] = "segmentation_report.txt") -> Dict:
    print("=" * 60)
    print("  ОЦЕНКА ТОЧНОСТИ СЕГМЕНТАЦИИ (3 класса)")
    print("=" * 60)

    print("\n[1/4] Загрузка изображений...")
    img_gt, img_pred = load_images(path_gt_img, path_pred_img)
    print(f"  GT: {img_gt.shape}, Pred: {img_pred.shape}")

    print("\n[2/4] Извлечение масок по цветам...")
    masks_gt = extract_color_masks(img_gt)
    masks_pred = extract_color_masks(img_pred)
    for name in CLASS_NAMES:
        print(f"  [{COLOR_RANGES[name]['label']}] "
              f"GT={masks_gt[name].sum() // 255} px, "
              f"Pred={masks_pred[name].sum() // 255} px")

    print("\n[3/4] Вычисление метрик по каждому классу...")
    per_class: Dict[str, Dict[str, float]] = {}
    for name in CLASS_NAMES:
        metrics = compute_metrics_for_class(
            masks_gt[name], masks_pred[name], hausdorff_percentile
        )
        per_class[name] = metrics
        label = COLOR_RANGES[name]['label']
        print(f"\n  ▶ {label}:")
        print(f"      IoU           : {metrics['IoU']:.4f}")
        print(f"      Hausdorff p95 : {metrics['Hausdorff_px']:.2f} px")
        print(f"      F1-score      : {metrics['F1']:.4f}")
        print(f"      AUC-ROC       : {metrics['AUC']:.4f}")
        print(f"      Precision     : {metrics['Precision']:.4f}")
        print(f"      Recall        : {metrics['Recall']:.4f}")

    mean_metrics = {
        "IoU_mean": np.mean([per_class[n]["IoU"] for n in CLASS_NAMES]),
        "Hausdorff_mean": np.mean([per_class[n]["Hausdorff_px"] for n in CLASS_NAMES]),
        "F1_mean": np.mean([per_class[n]["F1"] for n in CLASS_NAMES]),
        "AUC_mean": np.mean([per_class[n]["AUC"] for n in CLASS_NAMES]),
        "Precision_mean": np.mean([per_class[n]["Precision"] for n in CLASS_NAMES]),
        "Recall_mean": np.mean([per_class[n]["Recall"] for n in CLASS_NAMES]),
    }
    print("\n  ▶ СРЕДНИЕ (macro):")
    for k, v in mean_metrics.items():
        print(f"      {k:18s}: {v:.4f}")

    combined_gt = extract_combined_mask(masks_gt)
    combined_pred = extract_combined_mask(masks_pred)
    combined_metrics = compute_metrics_for_class(
        combined_gt, combined_pred, hausdorff_percentile
    )
    print("\n  ▶ ОБЩАЯ МАСКА (все классы объединены):")
    print(f"      IoU           : {combined_metrics['IoU']:.4f}")
    print(f"      Hausdorff p95 : {combined_metrics['Hausdorff_px']:.2f} px")
    print(f"      F1-score      : {combined_metrics['F1']:.4f}")
    print(f"      AUC-ROC       : {combined_metrics['AUC']:.4f}")

    if show_visualization:
        print("\n[4/4] Визуализация...")
        _visualize(img_gt, img_pred, masks_gt, masks_pred,
                   per_class, mean_metrics, combined_metrics,
                   save_path=save_viz_path)

    results = {
        "per_class": per_class,
        "mean": mean_metrics,
        "combined": combined_metrics,
    }
    if save_report:
        with open(save_report, "w", encoding="utf-8") as f:
            f.write("=== ОТЧЁТ ОЦЕНКИ СЕГМЕНТАЦИИ (3 класса) ===\n")
            for name in CLASS_NAMES:
                f.write(f"\n[{COLOR_RANGES[name]['label']}]\n")
                for k, v in per_class[name].items():
                    f.write(f"  {k}: {v}\n")
            f.write("\n[MACRO MEAN]\n")
            for k, v in mean_metrics.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n[COMBINED]\n")
            for k, v in combined_metrics.items():
                f.write(f"  {k}: {v}\n")
        print(f"\nОтчёт сохранён: {save_report}")

    print("\n" + "=" * 60)
    return results


# ============================================================
# 7. ВИЗУАЛИЗАЦИЯ (ПЕРЕПИСАНА — подписи всегда читаемы)
# ============================================================
def _make_caption(metrics: Dict[str, float],
                  label: str,
                  prefix: str = "") -> str:
    """Формирует многострочную подпись для subplot'а."""
    return (
        f"{prefix}{label}\n"
        f"IoU={metrics['IoU']:.3f}  •  F1={metrics['F1']:.3f}\n"
        f"Hausd={metrics['Hausdorff_px']:.1f}px  •  AUC={metrics['AUC']:.3f}\n"
        f"Prec={metrics['Precision']:.3f}  •  Recall={metrics['Recall']:.3f}"
    )


def _add_caption(ax, caption: str, color: str = "black"):
    """
    Добавляет подпись ПОД изображением с полупрозрачным фоном.
    Использует xlabel с bbox — текст всегда читается на любом фоне.
    """
    ax.set_xlabel(
        caption,
        fontsize=10,
        fontfamily="monospace",
        color=color,
        labelpad=8,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.92,
        ),
    )


def _visualize(img_gt, img_pred, masks_gt, masks_pred,
               per_class, mean_metrics, combined_metrics,
               save_path: Optional[str] = "segmentation_viz.png"):
    """
    Визуализация 3x3 + сводная таблица.
    Подписи вынесены под изображения (xlabel с bbox).
    """
    fig = plt.figure(figsize=(22, 20))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.25,
                          height_ratios=[1, 1, 1, 0.6])

    # ---------- Строка 1: исходные изображения и наложение ----------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    ax1.set_title("Ground Truth (цветная разметка)", fontsize=13, weight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    ax2.set_title("Предсказание (цветная разметка)", fontsize=13, weight="bold")
    ax2.axis("off")

    # Наложение: GT — контур, Pred — полупрозрачная заливка
    overlay = img_pred.copy().astype(float)
    for name in CLASS_NAMES:
        color_bgr = np.array(COLOR_RANGES[name]["bgr"], dtype=float)
        cnt, _ = cv2.findContours(masks_gt[name], cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay.astype(np.uint8), cnt, -1,
                         tuple(int(c) for c in color_bgr), 3)
        mask_pred_c = masks_pred[name] > 0
        overlay[mask_pred_c] = 0.55 * color_bgr + 0.45 * overlay[mask_pred_c]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax3.set_title("Наложение (GT=контур, Pred=заливка)", fontsize=13, weight="bold")
    ax3.axis("off")

    # ---------- Строка 2: маски GT по классам ----------
    for i, name in enumerate(CLASS_NAMES):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(masks_gt[name], cmap="gray")
        ax.set_title(f"GT: {COLOR_RANGES[name]['label']}",
                     fontsize=12, weight="bold", color=COLOR_RANGES[name]["mpl"])
        ax.axis("off")
        _add_caption(ax, _make_caption(per_class[name],
                                       COLOR_RANGES[name]['label'],
                                       prefix="GT: "))

    # ---------- Строка 3: маски Pred по классам ----------
    for i, name in enumerate(CLASS_NAMES):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(masks_pred[name], cmap="gray")
        ax.set_title(f"Pred: {COLOR_RANGES[name]['label']}",
                     fontsize=12, weight="bold", color=COLOR_RANGES[name]["mpl"])
        ax.axis("off")
        _add_caption(ax, _make_caption(per_class[name],
                                       COLOR_RANGES[name]['label'],
                                       prefix="Pred: "))

    # ---------- Строка 4: сводная таблица метрик ----------
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis("off")
    ax_table.set_title("Сводная таблица метрик", fontsize=14, weight="bold", pad=15)

    rows = [["Класс", "IoU", "Hausdorff\n(px)", "F1", "AUC",
             "Precision", "Recall", "TP", "FP", "TN", "FN"]]
    for name in CLASS_NAMES:
        m = per_class[name]
        rows.append([
            COLOR_RANGES[name]['label'],
            f"{m['IoU']:.4f}",
            f"{m['Hausdorff_px']:.2f}",
            f"{m['F1']:.4f}",
            f"{m['AUC']:.4f}",
            f"{m['Precision']:.4f}",
            f"{m['Recall']:.4f}",
            str(m["TP"]), str(m["FP"]), str(m["TN"]), str(m["FN"]),
        ])
    rows.append(["MACRO MEAN",
                 f"{mean_metrics['IoU_mean']:.4f}",
                 f"{mean_metrics['Hausdorff_mean']:.2f}",
                 f"{mean_metrics['F1_mean']:.4f}",
                 f"{mean_metrics['AUC_mean']:.4f}",
                 f"{mean_metrics['Precision_mean']:.4f}",
                 f"{mean_metrics['Recall_mean']:.4f}",
                 "", "", "", ""])
    rows.append(["COMBINED (все классы)",
                 f"{combined_metrics['IoU']:.4f}",
                 f"{combined_metrics['Hausdorff_px']:.2f}",
                 f"{combined_metrics['F1']:.4f}",
                 f"{combined_metrics['AUC']:.4f}",
                 f"{combined_metrics['Precision']:.4f}",
                 f"{combined_metrics['Recall']:.4f}",
                 str(combined_metrics["TP"]),
                 str(combined_metrics["FP"]),
                 str(combined_metrics["TN"]),
                 str(combined_metrics["FN"])])

    table = ax_table.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Стилизация шапки
    for j in range(len(rows[0])):
        table[0, j].set_facecolor("#2C3E50")
        table[0, j].set_text_props(color="white", weight="bold", fontsize=10)

    # Стилизация строки MACRO MEAN
    for j in range(len(rows[0])):
        table[len(rows) - 2, j].set_facecolor("#D5F5E3")
        table[len(rows) - 2, j].set_text_props(weight="bold")

    # Стилизация строки COMBINED
    for j in range(len(rows[0])):
        table[len(rows) - 1, j].set_facecolor("#FADBD8")
        table[len(rows) - 1, j].set_text_props(weight="bold")

    # Чередование цветов строк
    for i in range(1, len(rows) - 2):
        for j in range(len(rows[0])):
            if i % 2 == 0:
                table[i, j].set_facecolor("#F8F9F9")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"  Визуализация сохранена: {save_path}")

    plt.close(fig)


# ============================================================
# 8. ОЦЕНКА ПО ГОТОВЫМ МАСКАМ
# ============================================================
def evaluate_from_masks(masks_gt: Dict[str, np.ndarray],
                        masks_pred: Dict[str, np.ndarray],
                        hausdorff_percentile: int = 95) -> Dict:
    per_class = {}
    for name in masks_gt.keys():
        per_class[name] = compute_metrics_for_class(
            masks_gt[name], masks_pred[name], hausdorff_percentile
        )
    mean_metrics = {
        "IoU_mean": np.mean([per_class[n]["IoU"] for n in per_class]),
        "Hausdorff_mean": np.mean([per_class[n]["Hausdorff_px"] for n in per_class]),
        "F1_mean": np.mean([per_class[n]["F1"] for n in per_class]),
        "AUC_mean": np.mean([per_class[n]["AUC"] for n in per_class]),
    }
    return {"per_class": per_class, "mean": mean_metrics}


# ============================================================
# 9. MAIN
# ============================================================
if __name__ == "__main__":
    PATH_GT_IMG = "path_gt.JPG"
    PATH_PRED_IMG = "path_pred.JPG"
    SAVE_REPORT = "segmentation_report.txt"
    SAVE_VIZ = "segmentation_viz.png"

    if not os.path.exists(PATH_GT_IMG):
        print(f"⚠ Файл не найден: {PATH_GT_IMG}")
        print("  Укажите корректные пути в переменных PATH_GT_IMG и PATH_PRED_IMG")
    else:
        results = evaluate_segmentation(
            path_gt_img=PATH_GT_IMG,
            path_pred_img=PATH_PRED_IMG,
            hausdorff_percentile=95,
            show_visualization=True,
            save_viz_path=SAVE_VIZ,
            save_report=SAVE_REPORT,
        )
        print("\nИтоговые средние метрики:")
        for k, v in results["mean"].items():
            print(f"  {k:20s}: {v:.4f}")