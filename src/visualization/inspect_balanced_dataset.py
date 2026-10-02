"""Xem kết quả dataset đã balance (tầng 2 — DeepSMOTE): báo cáo + ảnh thật vs synthetic.

Script này chỉ ĐỌC `balanced_train.npz` (tạo bởi `src/deepsmote/run_deepsmote.py`) —
không train, không sinh thêm ảnh. In bảng số mẫu thật/synthetic mỗi class và xuất
ảnh biểu đồ + lưới so sánh thật/synthetic.

Cách dùng:
    python src/visualization/inspect_balanced_dataset.py --dataset mnist --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp dạng script: cho phép import `preprocessing`/`visualization`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # CLI không cần hiển thị hình, chỉ lưu file (phải set trước khi import pyplot)

import matplotlib.pyplot as plt  # noqa: E402

from preprocessing.config import DATASET_CONFIGS, DEFAULT_SEED, PROCESSED_DIR  # noqa: E402
from visualization.deepsmote_report import (  # noqa: E402
    format_balance_report,
    load_balanced_dataset,
    most_imbalanced_class,
    plot_balance_distribution,
    plot_real_vs_synthetic,
)

logger = logging.getLogger("inspect_balanced_dataset")

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def inspect_balanced_dataset(
    key: str,
    seed: int = DEFAULT_SEED,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    class_index: int | None = None,
    samples_per_class: int = 6,
) -> Path:
    """In báo cáo + xuất ảnh cho dataset đã balance. Trả về thư mục chứa ảnh xuất ra."""
    ds = load_balanced_dataset(key, seed, processed_dir)
    class_index = most_imbalanced_class(ds) if class_index is None else class_index

    out_dir = reports_dir / ds.cfg.key / ds.cfg.imbalance_dirname / f"seed_{seed}" / "deepsmote"
    dist_path = out_dir / "balance_distribution.png"
    compare_path = out_dir / f"real_vs_synthetic_class{class_index}.png"
    plt.close(plot_balance_distribution(ds, dist_path))
    plt.close(plot_real_vs_synthetic(ds, class_index, n=samples_per_class, seed=seed, out_path=compare_path))

    print(f"\nDataset: {ds.cfg.display_name} (đã balance bằng DeepSMOTE)")
    print(f"Random seed: {seed}")
    print(f"Method: {ds.metadata.get('method', '?')}")
    print()
    print(format_balance_report(ds))
    print()
    print(f"Biểu đồ phân bố (thật + synthetic): {dist_path}")
    print(f"So sánh thật vs synthetic (class {class_index}): {compare_path}")
    print()

    return out_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed của dataset đã tạo (mặc định 42)")
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    p.add_argument("--class-index", type=int, default=None,
                   help="class để vẽ lưới so sánh thật/synthetic (mặc định: class hiếm nhất)")
    p.add_argument("--samples-per-class", type=int, default=6)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args(argv)
    try:
        inspect_balanced_dataset(
            args.dataset, seed=args.seed, processed_dir=args.processed_dir, reports_dir=args.reports_dir,
            class_index=args.class_index, samples_per_class=args.samples_per_class,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
