"""SMOTE kinh điển áp dụng trong latent space — đây là bước nội suy THẬT của DeepSMOTE.

Khác với loss "tái tạo thiên lệch theo class" lúc train autoencoder (không hề nội
suy, xem README), hàm ở đây mới thực sự nội suy tuyến tính giữa 2 vector latent
thật, đúng công thức SMOTE gốc — chỉ khác là áp dụng trên latent vector thay vì
trên pixel ảnh.
"""

from __future__ import annotations

import numpy as np


def _k_nearest_indices(latent: np.ndarray, k: int) -> np.ndarray:
    """Chỉ số k hàng xóm gần nhất (không tính chính nó) cho từng điểm, brute-force.

    Đủ nhanh vì mỗi class tối đa vài nghìn mẫu (ma trận khoảng cách n x n, n~4500).
    """
    diff = latent[:, None, :] - latent[None, :, :]
    dist2 = np.einsum("ijd,ijd->ij", diff, diff)
    np.fill_diagonal(dist2, np.inf)
    return np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]


def smote_in_latent_space(latent: np.ndarray, n_to_generate: int, k: int, seed: int) -> np.ndarray:
    """sample = base + rand(0,1) * (hàng_xóm - base), nội suy giữa 2 latent vector thật.

    `latent`: (n, n_z) — vector latent thật của 1 class (đã encode bằng encoder đã train,
    KHÔNG train thêm ở bước này). Trả về (n_to_generate, n_z) vector latent synthetic.
    """
    if len(latent) < 2:
        raise ValueError("Cần ít nhất 2 mẫu thật trong class để SMOTE được")
    k = min(k, len(latent) - 1)
    neighbor_idx = _k_nearest_indices(latent, k)

    rng = np.random.default_rng(seed)
    base_idx = rng.integers(0, len(latent), size=n_to_generate)
    neighbor_choice = rng.integers(0, k, size=n_to_generate)
    chosen_neighbor_idx = neighbor_idx[base_idx, neighbor_choice]

    alpha = rng.random((n_to_generate, 1)).astype(np.float32)
    base = latent[base_idx]
    neighbor = latent[chosen_neighbor_idx]
    return (base + alpha * (neighbor - base)).astype(np.float32)
