"""Autoencoder (Encoder/Decoder) dùng cho DeepSMOTE.

Kiến trúc CNN đơn giản, tổng quát cho mọi kích thước ảnh vuông (28x28 hay 32x32)
và mọi số kênh (1 hay 3): encoder dùng `AdaptiveAvgPool2d` nên không phụ thuộc
kích thước spatial sau 4 lớp stride-2, decoder nội suy về đúng kích thước ở bước
cuối — tránh phải tính tay kích thước qua từng lớp như code gốc của tác giả
(chỉ viết sẵn cho đúng 28x28).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels: int, dim_h: int, n_z: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_h, dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_h * 2, dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim_h * 4, dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim_h * 8, n_z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int, out_size: int, dim_h: int, n_z: int):
        super().__init__()
        self.dim_h = dim_h
        self.out_size = out_size
        self.fc = nn.Sequential(nn.Linear(n_z, dim_h * 8 * 4 * 4), nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(dim_h * 8, dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim_h * 4, dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim_h * 2, dim_h, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim_h),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_h, out_channels, 3, 1, 1),
            nn.Tanh(),  # ảnh output trong [-1, 1], khớp chuẩn hóa input ở data.py
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.dim_h * 8, 4, 4)
        x = self.deconv(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return x
