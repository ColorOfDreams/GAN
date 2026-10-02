"""Load dataset đã xử lý và so sánh với số liệu paper DeepSMOTE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from preprocessing.config import DATASET_CONFIGS, PROCESSED_DIR, DatasetConfig


@dataclass(frozen=True)
class LoadedDataset:
    cfg: DatasetConfig
    seed: int
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    metadata: dict


def load_processed_dataset(key: str, seed: int, processed_dir: Path = PROCESSED_DIR) -> LoadedDataset:
    """Đọc train.npz, test.npz, metadata.json của một dataset đã tạo."""
    cfg = DATASET_CONFIGS[key]
    out_dir = cfg.output_dir(seed, processed_dir)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Chưa thấy dataset đã xử lý tại {out_dir}. "
            f"Hãy chạy: python src/preprocessing/create_imbalanced_dataset.py --dataset {key} --seed {seed}"
        )

    with np.load(out_dir / "train.npz") as train, np.load(out_dir / "test.npz") as test:
        train_x, train_y = train["x"], train["y"]
        test_x, test_y = test["x"], test["y"]
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    return LoadedDataset(cfg=cfg, seed=seed, train_x=train_x, train_y=train_y,
                         test_x=test_x, test_y=test_y, metadata=metadata)


@dataclass(frozen=True)
class ClassComparisonRow:
    class_index: int
    class_name: str
    expected: int   # số mẫu quy định trong paper DeepSMOTE (DATASET_CONFIGS)
    actual: int     # số mẫu thực tế đọc từ train.npz
    match: bool


def compare_with_deepsmote(ds: LoadedDataset) -> list[ClassComparisonRow]:
    """So sánh số mẫu mỗi class thực tế với số mẫu paper DeepSMOTE quy định.

    `DATASET_CONFIGS` (src/preprocessing/config.py) lấy trực tiếp từ các mảng
    `imbal` trong paper / trong `DeepSMOTE/GenerateSamples.py`, nên đây chính
    là phép so sánh "có giống DeepSMOTE không".
    """
    actual_counts = np.bincount(ds.train_y, minlength=ds.cfg.num_classes)
    return [
        ClassComparisonRow(
            class_index=c,
            class_name=ds.cfg.class_names[c],
            expected=expected,
            actual=int(actual_counts[c]),
            match=int(actual_counts[c]) == expected,
        )
        for c, expected in enumerate(ds.cfg.train_class_counts)
    ]


def comparison_all_match(rows: list[ClassComparisonRow]) -> bool:
    return all(row.match for row in rows)


def format_comparison_table(rows: list[ClassComparisonRow]) -> str:
    header = f"{'Class':<16}{'Paper DeepSMOTE':>16}{'Thực tế':>12}{'Kết quả':>10}"
    lines = [header, "-" * len(header)]
    for row in rows:
        status = "KHỚP" if row.match else "SAI"
        lines.append(f"{row.class_index} ({row.class_name:<10}){row.expected:>14}{row.actual:>12}{status:>10}")
    return "\n".join(lines)


def plot_class_distribution(ds: LoadedDataset, rows: list[ClassComparisonRow], out_path: Path | None = None):
    """Biểu đồ cột: số mẫu mỗi class — paper DeepSMOTE vs. dataset thực tế đã tạo.

    Trả về `Figure` (không gọi `plt.close`), để dùng được cả từ CLI (lưu file
    rồi đóng) lẫn từ notebook (hiển thị inline). Chỉ lưu file khi có `out_path`.
    """
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.9), 4.5))
    ax.bar(x - width / 2, [r.expected for r in rows], width, label="Paper DeepSMOTE", color="#4C72B0")
    ax.bar(x + width / 2, [r.actual for r in rows], width, label="Dataset đã tạo", color="#DD8452")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.class_index}\n{r.class_name}" for r in rows], fontsize=8)
    ax.set_ylabel("Số mẫu (log scale)")
    ax.set_title(f"{ds.cfg.display_name} — phân bố class (seed={ds.seed}, tỉ lệ {ds.cfg.imbalance_ratio_str})")
    ax.legend()
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig


def plot_sample_grid(ds: LoadedDataset, out_path: Path | None = None, samples_per_class: int = 6, seed: int = 0):
    """Lưới ảnh mẫu: mỗi dòng là một class, để nhìn trực quan dữ liệu đã tạo.

    Trả về `Figure` (không gọi `plt.close`); chỉ lưu file khi có `out_path`.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    num_classes = ds.cfg.num_classes
    fig, axes = plt.subplots(num_classes, samples_per_class,
                             figsize=(samples_per_class * 1.2, num_classes * 1.2))

    for c in range(num_classes):
        idx_of_class = np.flatnonzero(ds.train_y == c)
        pick = rng.choice(idx_of_class, size=min(samples_per_class, len(idx_of_class)), replace=False)
        for col in range(samples_per_class):
            ax = axes[c, col]
            ax.axis("off")
            if col < len(pick):
                image = ds.train_x[pick[col]]
                ax.imshow(image.squeeze(-1) if image.shape[-1] == 1 else image,
                          cmap="gray" if image.shape[-1] == 1 else None)
            if col == 0:
                ax.text(-0.3, 0.5, f"{c}\n{ds.cfg.class_names[c]}", transform=ax.transAxes,
                        ha="right", va="center", fontsize=8)

    fig.suptitle(f"{ds.cfg.display_name} — mẫu ảnh theo class (seed={ds.seed})")
    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
    return fig
