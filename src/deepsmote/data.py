"""Đọc dataset mất cân bằng (tầng 1) thành tensor cho autoencoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from preprocessing.config import DATASET_CONFIGS, PROCESSED_DIR, DatasetConfig


def load_train_tensors(
    dataset_key: str, seed: int, processed_dir: Path = PROCESSED_DIR
) -> tuple[torch.Tensor, torch.Tensor, DatasetConfig, Path]:
    """Đọc `train.npz` (uint8 NHWC, [0,255]) và chuẩn hóa thành tensor NCHW float [-1, 1].

    [-1, 1] khớp với activation `Tanh` ở lớp cuối decoder. Trả về thêm `DatasetConfig`
    và thư mục chứa dataset, để nơi gọi biết lưu model/kết quả vào đâu.
    """
    cfg = DATASET_CONFIGS[dataset_key]
    out_dir = cfg.output_dir(seed, processed_dir)
    train_path = out_dir / "train.npz"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Chưa có dataset mất cân bằng tại {train_path}. Hãy chạy trước:\n"
            f"  python src/preprocessing/create_imbalanced_dataset.py --dataset {dataset_key} --seed {seed}"
        )

    with np.load(train_path) as data:
        x, y = data["x"], data["y"]

    images = torch.from_numpy(x).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    labels = torch.from_numpy(y).long()
    return images, labels, cfg, out_dir


def denormalize_to_uint8(images: torch.Tensor) -> np.ndarray:
    """Tensor NCHW [-1, 1] -> mảng NHWC uint8 [0, 255], nghịch đảo của chuẩn hóa ở trên."""
    images = ((images.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
    return images.permute(0, 2, 3, 1).cpu().numpy()
