"""Load dataset đã balance (tầng 2 — DeepSMOTE) và so sánh ảnh thật/synthetic.

Module này chỉ ĐỌC `balanced_train.npz` do `src/deepsmote/generate_balanced_dataset.py`
tạo ra — không train, không sinh thêm ảnh, không sửa dữ liệu.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preprocessing.config import DATASET_CONFIGS, PROCESSED_DIR, DatasetConfig


@dataclass(frozen=True)
class LoadedBalancedDataset:
    cfg: DatasetConfig
    seed: int
    x: np.ndarray             # uint8 NHWC — cả ảnh thật lẫn synthetic
    y: np.ndarray             # int64
    is_synthetic: np.ndarray  # bool — False = ảnh thật (từ tầng 1), True = DeepSMOTE sinh ra
    metadata: dict


def load_balanced_dataset(
    dataset_key: str, seed: int, processed_dir: Path = PROCESSED_DIR
) -> LoadedBalancedDataset:
    """Đọc `balanced_train.npz` + `balanced_metadata.json` của một dataset đã chạy tầng 2."""
    cfg = DATASET_CONFIGS[dataset_key]
    model_dir = cfg.output_dir(seed, processed_dir) / "deepsmote"
    balanced_path = model_dir / "balanced_train.npz"
    meta_path = model_dir / "balanced_metadata.json"
    if not balanced_path.exists():
        raise FileNotFoundError(
            f"Chưa có dataset đã balance tại {balanced_path}. Hãy chạy trước:\n"
            f"  python src/deepsmote/run_deepsmote.py --dataset {dataset_key} --seed {seed}"
        )

    with np.load(balanced_path) as data:
        x, y, is_synth = data["x"], data["y"], data["is_synthetic"]
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return LoadedBalancedDataset(cfg=cfg, seed=seed, x=x, y=y, is_synthetic=is_synth, metadata=metadata)


def format_balance_report(ds: LoadedBalancedDataset) -> str:
    """Bảng số mẫu thật/synthetic mỗi class — cho thấy rõ DeepSMOTE đã bù bao nhiêu."""
    header = f"{'Class':<16}{'Thật':>10}{'Synthetic':>12}{'Tổng':>10}"
    lines = [header, "-" * len(header)]
    for c in range(ds.cfg.num_classes):
        real = int(np.sum((ds.y == c) & ~ds.is_synthetic))
        synth = int(np.sum((ds.y == c) & ds.is_synthetic))
        lines.append(f"{c} ({ds.cfg.class_names[c]:<10}){real:>12}{synth:>12}{real + synth:>10}")
    lines.append("-" * len(header))
    total_real, total_synth = int((~ds.is_synthetic).sum()), int(ds.is_synthetic.sum())
    lines.append(f"{'Tổng':<16}{total_real:>12}{total_synth:>12}{total_real + total_synth:>10}")
    return "\n".join(lines)


def most_imbalanced_class(ds: LoadedBalancedDataset) -> int:
    """Class có nhiều ảnh synthetic nhất — tức class hiếm nhất ở tầng 1, demo trực quan nhất."""
    synth_counts = [int(np.sum((ds.y == c) & ds.is_synthetic)) for c in range(ds.cfg.num_classes)]
    return int(np.argmax(synth_counts))


def plot_real_vs_synthetic(
    ds: LoadedBalancedDataset, class_index: int, n: int = 6, seed: int = 0, out_path: Path | None = None
):
    """Lưới 2 hàng: ảnh thật (trên) vs ảnh synthetic (dưới) của 1 class — so sánh trực quan chất lượng."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    real_idx = np.flatnonzero((ds.y == class_index) & ~ds.is_synthetic)
    synth_idx = np.flatnonzero((ds.y == class_index) & ds.is_synthetic)
    if len(synth_idx) == 0:
        raise ValueError(f"Class {class_index} không có ảnh synthetic nào (đã đủ số mẫu thật ở tầng 1)")

    n_real = min(n, len(real_idx))
    n_synth = min(n, len(synth_idx))
    pick_real = rng.choice(real_idx, size=n_real, replace=False)
    pick_synth = rng.choice(synth_idx, size=n_synth, replace=False)
    n_cols = max(n_real, n_synth)

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 1.3, 3))
    if n_cols == 1:
        axes = axes.reshape(2, 1)
    for row, (pick, row_label) in enumerate([(pick_real, "Thật"), (pick_synth, "Synthetic")]):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(pick):
                img = ds.x[pick[col]]
                ax.imshow(img.squeeze(-1) if img.shape[-1] == 1 else img,
                          cmap="gray" if img.shape[-1] == 1 else None)
            if col == 0:
                ax.text(-0.3, 0.5, row_label, transform=ax.transAxes, ha="right", va="center", fontsize=9)

    fig.suptitle(f"{ds.cfg.display_name} — class {class_index} ({ds.cfg.class_names[class_index]}): "
                f"thật vs synthetic")
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig


def plot_balance_distribution(ds: LoadedBalancedDataset, out_path: Path | None = None):
    """Biểu đồ cột chồng: phần thật + phần synthetic mỗi class — thấy rõ đã balance tới đâu."""
    import matplotlib.pyplot as plt

    real_counts = [int(np.sum((ds.y == c) & ~ds.is_synthetic)) for c in range(ds.cfg.num_classes)]
    synth_counts = [int(np.sum((ds.y == c) & ds.is_synthetic)) for c in range(ds.cfg.num_classes)]
    x = np.arange(ds.cfg.num_classes)

    fig, ax = plt.subplots(figsize=(max(6, ds.cfg.num_classes * 0.9), 4.5))
    ax.bar(x, real_counts, label="Thật (tầng 1)", color="#4C72B0")
    ax.bar(x, synth_counts, bottom=real_counts, label="Synthetic (DeepSMOTE)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n{ds.cfg.class_names[c]}" for c in range(ds.cfg.num_classes)], fontsize=8)
    ax.set_ylabel("Số mẫu")
    ax.set_title(f"{ds.cfg.display_name} — dataset đã balance (seed={ds.seed})")
    ax.legend()
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig
