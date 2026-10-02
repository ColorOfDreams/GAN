"""Chạy toàn bộ DeepSMOTE tầng 2: train autoencoder rồi sinh dataset đã balance.

Cách dùng:
    python src/deepsmote/run_deepsmote.py --dataset mnist --seed 42 --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp dạng script: cho phép import `preprocessing`/`deepsmote`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepsmote.config import DEFAULT_TRAIN_CONFIG, TrainConfig  # noqa: E402
from deepsmote.generate_balanced_dataset import generate_balanced_dataset  # noqa: E402
from deepsmote.train_autoencoder import train_autoencoder  # noqa: E402
from preprocessing.config import DATASET_CONFIGS, DEFAULT_SEED, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger("run_deepsmote")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--epochs", type=int, default=DEFAULT_TRAIN_CONFIG.epochs)
    p.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG.batch_size)
    p.add_argument("--dim-h", type=int, default=DEFAULT_TRAIN_CONFIG.dim_h)
    p.add_argument("--n-z", type=int, default=DEFAULT_TRAIN_CONFIG.n_z)
    p.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG.lr)
    p.add_argument("--k-neighbors", type=int, default=DEFAULT_TRAIN_CONFIG.k_neighbors)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args(argv)
    cfg = TrainConfig(dim_h=args.dim_h, n_z=args.n_z, lr=args.lr, epochs=args.epochs,
                      batch_size=args.batch_size, k_neighbors=args.k_neighbors)

    train_autoencoder(args.dataset, seed=args.seed, processed_dir=args.processed_dir, cfg=cfg)
    generate_balanced_dataset(args.dataset, seed=args.seed, processed_dir=args.processed_dir,
                              k_neighbors=cfg.k_neighbors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
