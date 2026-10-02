"""Xem kết quả và so sánh dataset đã tạo với số liệu paper DeepSMOTE.

Script này chỉ ĐỌC dataset đã có trong `data/processed/` (tạo bởi
`src/preprocessing/create_imbalanced_dataset.py`) — không sample lại, không
sửa dữ liệu. Nó in bảng so sánh số mẫu mỗi class với `DATASET_CONFIGS`
(tức số liệu paper DeepSMOTE) và xuất ảnh biểu đồ + lưới mẫu ảnh.

Cách dùng:
    python src/visualization/inspect_dataset.py --dataset mnist --seed 42
    python src/visualization/inspect_dataset.py --dataset all
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
from visualization.report import (  # noqa: E402
    compare_with_deepsmote,
    comparison_all_match,
    format_comparison_table,
    load_processed_dataset,
    plot_class_distribution,
    plot_sample_grid,
)

logger = logging.getLogger("inspect_dataset")

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def inspect_dataset(
    key: str,
    seed: int = DEFAULT_SEED,
    processed_dir: Path = PROCESSED_DIR,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    samples_per_class: int = 6,
) -> bool:
    """In báo cáo so sánh + xuất ảnh cho một dataset. Trả về True nếu mọi class khớp paper."""
    ds = load_processed_dataset(key, seed, processed_dir)
    rows = compare_with_deepsmote(ds)
    all_match = comparison_all_match(rows)

    out_dir = reports_dir / ds.cfg.key / ds.cfg.imbalance_dirname / f"seed_{seed}"
    dist_path = out_dir / "class_distribution.png"
    grid_path = out_dir / "sample_grid.png"
    plt.close(plot_class_distribution(ds, rows, dist_path))
    plt.close(plot_sample_grid(ds, grid_path, samples_per_class=samples_per_class, seed=seed))

    print(f"\nDataset: {ds.cfg.display_name}")
    print(f"Random seed: {seed}")
    print(f"Imbalance ratio (paper DeepSMOTE): {ds.cfg.imbalance_ratio_str}")
    print()
    print(format_comparison_table(rows))
    print()
    print(f"Tổng số mẫu train: {sum(r.actual for r in rows)} (paper: {sum(r.expected for r in rows)})")
    print(f"Test set: {len(ds.test_y)} mẫu (giữ nguyên, không resample)")
    print()
    print(f"Giống paper DeepSMOTE: {'CÓ' if all_match else 'KHÔNG'}")
    print(f"Biểu đồ phân bố class: {dist_path}")
    print(f"Lưới mẫu ảnh: {grid_path}")
    print()

    return all_match


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=[*DATASET_CONFIGS, "all"],
                        help="dataset cần xem, hoặc 'all' để xem tất cả")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed của dataset đã tạo (mặc định 42)")
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR,
                        help="thư mục chứa dataset đã xử lý (mặc định data/processed)")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR,
                        help="thư mục xuất báo cáo/ảnh (mặc định reports/)")
    parser.add_argument("--samples-per-class", type=int, default=6, help="số ảnh mẫu hiển thị mỗi class")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args(argv)
    keys = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]

    results = {}
    for key in keys:
        try:
            results[key] = inspect_dataset(
                key, seed=args.seed, processed_dir=args.processed_dir,
                reports_dir=args.reports_dir, samples_per_class=args.samples_per_class,
            )
        except FileNotFoundError as exc:
            logger.error(str(exc))
            results[key] = False

    mismatched = [key for key, ok in results.items() if not ok]
    if mismatched:
        logger.error("Không khớp paper DeepSMOTE (hoặc chưa tạo): %s", ", ".join(mismatched))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
