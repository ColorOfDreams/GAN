"""Cấu hình train autoencoder + sinh mẫu DeepSMOTE (tầng 2)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    dim_h: int = 64              # độ rộng kênh cơ sở của encoder/decoder (giống paper)
    n_z: int = 128                # số chiều latent space
    lr: float = 2e-4
    epochs: int = 50              # paper dùng 200; mặc định thấp hơn để chạy thử nhanh, override bằng --epochs
    batch_size: int = 128
    k_neighbors: int = 5           # số hàng xóm dùng khi SMOTE trong latent space (không tính chính nó)
    biased_samples_per_step: int = 100  # số mẫu lấy mỗi bước cho loss tái tạo thiên lệch theo class


DEFAULT_TRAIN_CONFIG = TrainConfig()
