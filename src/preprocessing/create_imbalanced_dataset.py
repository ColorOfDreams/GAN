"""Tạo training set mất cân bằng theo protocol dữ liệu của DeepSMOTE.

Cách dùng:
    python src/preprocessing/create_imbalanced_dataset.py --dataset mnist --seed 42
    python src/preprocessing/create_imbalanced_dataset.py --dataset all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

if __package__ in (None, ""):  # chạy trực tiếp dạng script: cho phép import `preprocessing`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocessing.config import (  # noqa: E402
    DATASET_CONFIGS,
    DEFAULT_SEED,
    PAPER_REFERENCE,
    PROCESSED_DIR,
    RAW_DIR,
    DatasetConfig,
)
from preprocessing.loaders import load_raw_dataset  # noqa: E402
from preprocessing.sampling import sample_imbalanced_indices  # noqa: E402
from preprocessing.validation import CheckResult, validate_dataset  # noqa: E402

logger = logging.getLogger("create_imbalanced_dataset")

TRAIN_FILE = "train.npz"
TEST_FILE = "test.npz"
METADATA_FILE = "metadata.json"
STYLEGAN_DIR = "stylegan2_ada"


def create_imbalanced_dataset(
    key: str,
    seed: int = DEFAULT_SEED,
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    download: bool = True,
    export_png: bool = False,
) -> bool:
    """Tạo, lưu và kiểm tra một dataset mất cân bằng. Trả về True nếu validation đạt."""
    cfg = DATASET_CONFIGS[key]
    out_dir = cfg.output_dir(seed, processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_dataset(key, raw_dir, download)
    logger.info("Train gốc: %d mẫu, test gốc: %d mẫu", len(raw.train), len(raw.test))

    source_indices = sample_imbalanced_indices(raw.train.labels, cfg.train_class_counts, seed)
    logger.info("Đã sample %d index training với seed %d", len(source_indices), seed)

    save_split(out_dir / TRAIN_FILE, x=raw.train.fetch(source_indices),
               y=raw.train.labels[source_indices], source_indices=source_indices)
    save_split(out_dir / TEST_FILE, x=raw.test.all_images(), y=raw.test.labels)

    # Kiểm tra dữ liệu thực sự nằm trên đĩa, không phải bản sao trong bộ nhớ.
    train, test = load_split(out_dir / TRAIN_FILE), load_split(out_dir / TEST_FILE)
    checks = validate_dataset(cfg, raw, seed, train["x"], train["y"], train["source_indices"],
                              test["x"], test["y"])
    passed = all(c.passed for c in checks)

    write_metadata(out_dir / METADATA_FILE, cfg, seed, train, test, checks)
    if export_png:
        export_stylegan_folder(out_dir / STYLEGAN_DIR, train["x"], train["y"])

    print_summary(cfg, seed, train["y"], checks, out_dir)
    return passed


def save_split(path: Path, **arrays: np.ndarray) -> None:
    np.savez_compressed(path, **arrays)
    logger.info("Đã lưu %s", path)


def load_split(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


def _sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _class_counts(cfg: DatasetConfig, labels: np.ndarray) -> dict[str, int]:
    counts = np.bincount(labels, minlength=cfg.num_classes)
    return {str(c): int(counts[c]) for c in range(cfg.num_classes)}


def write_metadata(path: Path, cfg: DatasetConfig, seed: int, train: dict, test: dict,
                   checks: list[CheckResult]) -> None:
    metadata = {
        "dataset": cfg.display_name,
        "dataset_key": cfg.key,
        "protocol": PAPER_REFERENCE,
        "seed": seed,
        "class_names": {str(c): name for c, name in enumerate(cfg.class_names)},
        "class_counts": _class_counts(cfg, train["y"]),
        "total_train": int(len(train["y"])),
        "imbalance_ratio": cfg.imbalance_ratio_str,
        "imbalance_ratio_value": cfg.imbalance_ratio,
        "image_format": {"shape_hwc": list(cfg.image_shape), "layout": "NHWC", "dtype": "uint8",
                         "pixel_range": [0, 255], "normalized": False},
        "train": {
            "file": TRAIN_FILE,
            "arrays": {"x": "ảnh", "y": "nhãn", "source_indices": "index trong train split gốc"},
            "sampling": "ngẫu nhiên đều theo từng class, không hoàn lại, từ train split gốc",
            "sha256_source_indices": _sha256(train["source_indices"]),
            "sha256_images": _sha256(train["x"]),
        },
        "test": {
            "file": TEST_FILE,
            "description": "test split gốc, không resample",
            "size": int(len(test["y"])),
            "class_counts": _class_counts(cfg, test["y"]),
            "sha256": _sha256(test["x"], test["y"]),
        },
        "validation": {"passed": all(c.passed for c in checks), "checks": [c.to_dict() for c in checks]},
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "numpy_version": np.__version__,
    }
    path.write_text(json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8")
    logger.info("Đã lưu %s", path)


def export_stylegan_folder(folder: Path, images: np.ndarray, labels: np.ndarray) -> None:
    """Ghi PNG + dataset.json theo layout mà dataset_tool.py của StyleGAN2-ADA chấp nhận."""
    from PIL import Image

    entries = []
    for i, (image, label) in enumerate(zip(images, labels)):
        rel_path = f"{int(label)}/img{i:06d}.png"
        (folder / str(int(label))).mkdir(parents=True, exist_ok=True)
        Image.fromarray(image.squeeze(-1) if image.shape[-1] == 1 else image).save(folder / rel_path)
        entries.append([rel_path, int(label)])
    (folder / "dataset.json").write_text(json.dumps({"labels": entries}), encoding="utf-8")
    logger.info("Đã xuất %d ảnh PNG cho StyleGAN2-ADA vào %s", len(entries), folder)


def print_summary(cfg: DatasetConfig, seed: int, labels: np.ndarray, checks: list[CheckResult],
                  out_dir: Path) -> None:
    lines = [
        "",
        f"Dataset: {cfg.display_name}",
        f"Random seed: {seed}",
        "",
        "Class distribution:",
        *(f"Class {c}: {n}" for c, n in _class_counts(cfg, labels).items()),
        "",
        f"Total: {len(labels)}",
        f"Imbalance Ratio: {cfg.imbalance_ratio_str}",
        "",
        "Các bước kiểm tra:",
        *(f"  [{'OK' if c.passed else 'LỖI'}] {c.name}: {c.detail}" for c in checks),
        "",
        f"Validation: {'PASSED' if all(c.passed for c in checks) else 'FAILED'}",
        f"Output: {out_dir}",
        "",
    ]
    print("\n".join(lines))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=[*DATASET_CONFIGS, "all"],
                        help="dataset cần tạo, hoặc 'all' để tạo tất cả")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed (mặc định 42)")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="thư mục chứa dataset gốc")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR, help="thư mục gốc cho dataset đã xử lý")
    parser.add_argument("--no-download", action="store_true", help="báo lỗi thay vì tải dữ liệu còn thiếu")
    parser.add_argument("--export-png", action="store_true",
                        help="ghi thêm PNG + dataset.json cho dataset_tool.py của StyleGAN2-ADA")
    return parser.parse_args(argv)


def _use_utf8_console() -> None:
    """Tránh UnicodeEncodeError khi in tiếng Việt ra console/pipe cp1252 trên Windows."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _use_utf8_console()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args(argv)
    keys = list(DATASET_CONFIGS) if args.dataset == "all" else [args.dataset]

    results = {}
    for key in keys:
        try:
            results[key] = create_imbalanced_dataset(
                key, seed=args.seed, raw_dir=args.raw_dir, processed_dir=args.output_dir,
                download=not args.no_download, export_png=args.export_png,
            )
        except Exception:
            logger.exception("Tạo %s thất bại", key)
            results[key] = False

    failed = [key for key, ok in results.items() if not ok]
    if failed:
        logger.error("Validation FAILED cho: %s", ", ".join(failed))
        return 1
    logger.info("Tất cả dataset đều đạt validation: %s", ", ".join(keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
