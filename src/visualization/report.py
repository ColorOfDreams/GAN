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
    """Số mẫu mỗi class, paper vs. thực tế. Không in cột "khớp/sai" khi mọi số đều đúng —
    hai cột số bằng nhau đã đủ tự nói lên điều đó. Chỉ khi lệch mới in cảnh báo rõ ràng.
    """
    header = f"{'Class':<16}{'Paper DeepSMOTE':>16}{'Thực tế':>12}"
    lines = [header, "-" * len(header)]
    for row in rows:
        line = f"{row.class_index} ({row.class_name:<10}){row.expected:>14}{row.actual:>12}"
        if not row.match:
            line += f"   ⚠ LỆCH (mong đợi {row.expected}, thực tế {row.actual})"
        lines.append(line)
    return "\n".join(lines)


@dataclass(frozen=True)
class DatasetSummary:
    key: str
    display_name: str
    num_classes: int
    total_train: int
    imbalance_ratio: str
    all_match: bool


def summarize_dataset(key: str, seed: int, processed_dir: Path = PROCESSED_DIR) -> DatasetSummary | None:
    """Tóm tắt 1 dataset đã tạo: tổng mẫu train, tỉ lệ mất cân bằng, có khớp paper không.

    Trả về None nếu dataset chưa được tạo (thay vì ném lỗi), để dùng được ngay trong vòng lặp
    liệt kê nhiều dataset mà không cần try/except ở nơi gọi.
    """
    try:
        ds = load_processed_dataset(key, seed, processed_dir)
    except FileNotFoundError:
        return None
    rows = compare_with_deepsmote(ds)
    return DatasetSummary(
        key=key,
        display_name=ds.cfg.display_name,
        num_classes=ds.cfg.num_classes,
        total_train=sum(r.actual for r in rows),
        imbalance_ratio=ds.cfg.imbalance_ratio_str,
        all_match=comparison_all_match(rows),
    )


def format_overview_table(summaries: dict[str, DatasetSummary | None]) -> str:
    """Bảng tổng quan nhiều dataset: số liệu thật (tổng mẫu, tỉ lệ) thay vì chỉ CÓ/KHÔNG.

    Dataset chưa tạo hiện "(chưa tạo)" thay vì bị bỏ qua lặng lẽ. Dataset có class lệch so với
    paper được đánh dấu rõ — còn lại không cần nhãn gì thêm.
    """
    header = f"{'Dataset':<16}{'Số class':>10}{'Tổng mẫu train':>16}{'Tỉ lệ mất cân bằng':>22}"
    lines = [header, "-" * len(header)]
    for key, s in summaries.items():
        if s is None:
            lines.append(f"{key:<16}{'—':>10}{'—':>16}{'(chưa tạo)':>22}")
            continue
        line = f"{s.display_name:<16}{s.num_classes:>10}{s.total_train:>16}{s.imbalance_ratio:>22}"
        if not s.all_match:
            line += "   ⚠ LỆCH so với paper"
        lines.append(line)
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
