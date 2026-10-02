"""Undersampling ngẫu nhiên theo từng class (không tạo mẫu tổng hợp, không trùng lặp)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def sample_imbalanced_indices(
    labels: np.ndarray, class_counts: Sequence[int], seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for cls, count in enumerate(class_counts):
        pool = np.flatnonzero(labels == cls)
        if count > len(pool):
            raise ValueError(f"class {cls}: cần {count} mẫu nhưng chỉ có {len(pool)}")
        selected.append(np.sort(rng.choice(pool, size=count, replace=False)))
    return np.concatenate(selected).astype(np.int64)
