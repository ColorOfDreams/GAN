"""Cấu hình mất cân bằng (imbalance) theo paper DeepSMOTE.

Tham khảo:
    Dablain, Krawczyk, Chawla. "DeepSMOTE: Fusing Deep Learning and SMOTE for
    Imbalanced Data", IEEE TNNLS 2022. arXiv:2105.02340.

Mọi số lượng mẫu theo class dùng trong pipeline đều lấy từ DATASET_CONFIGS;
không nơi nào khác được hard-code các con số này.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

PAPER_REFERENCE = "DeepSMOTE (Dablain, Krawczyk, Chawla), arXiv:2105.02340"

# Số mẫu train mỗi class theo paper: class i -> counts[i].
MNIST_LIKE_COUNTS = (4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40)
CIFAR_LIKE_COUNTS = (4500, 2000, 1000, 800, 600, 500, 400, 250, 150, 80)
CELEBA_COUNTS = (9000, 4500, 1000, 500, 160)

# CelebA: paper dùng 5 thuộc tính màu tóc làm class, ảnh resize về 3x32x32.
CELEBA_HAIR_ATTRIBUTES = ("Black_Hair", "Brown_Hair", "Blond_Hair", "Gray_Hair", "Bald")
CELEBA_IMAGE_SIZE = 32


@dataclass(frozen=True)
class DatasetConfig:
    key: str                      # tên dùng trong CLI và tên thư mục output
    display_name: str
    class_names: tuple[str, ...]
    train_class_counts: tuple[int, ...]
    image_shape: tuple[int, int, int]  # (H, W, C), uint8
    original_test_size: int | None     # None khi test set được suy ra (CelebA)

    def __post_init__(self) -> None:
        if len(self.class_names) != len(self.train_class_counts):
            raise ValueError(f"{self.key}: class_names và train_class_counts khác độ dài")

    @property
    def num_classes(self) -> int:
        return len(self.train_class_counts)

    @property
    def total_train(self) -> int:
        return sum(self.train_class_counts)

    @property
    def imbalance_ratio(self) -> float:
        return max(self.train_class_counts) / min(self.train_class_counts)

    @property
    def imbalance_ratio_str(self) -> str:
        return f"{_format_number(self.imbalance_ratio)}:1"

    @property
    def imbalance_dirname(self) -> str:
        return f"imbalance_{_format_number(self.imbalance_ratio)}"

    def output_dir(self, seed: int, processed_dir: Path = PROCESSED_DIR) -> Path:
        return processed_dir / self.key / self.imbalance_dirname / f"seed_{seed}"


def _format_number(value: float) -> str:
    """100.0 -> '100', 56.25 -> '56.25'."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


_DIGITS = tuple(str(i) for i in range(10))

DATASET_CONFIGS: dict[str, DatasetConfig] = {
    cfg.key: cfg
    for cfg in (
        DatasetConfig(
            key="mnist",
            display_name="MNIST",
            class_names=_DIGITS,
            train_class_counts=MNIST_LIKE_COUNTS,
            image_shape=(28, 28, 1),
            original_test_size=10_000,
        ),
        DatasetConfig(
            key="fashion_mnist",
            display_name="Fashion-MNIST",
            class_names=(
                "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
            ),
            train_class_counts=MNIST_LIKE_COUNTS,
            image_shape=(28, 28, 1),
            original_test_size=10_000,
        ),
        DatasetConfig(
            key="cifar10",
            display_name="CIFAR-10",
            class_names=(
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ),
            train_class_counts=CIFAR_LIKE_COUNTS,
            image_shape=(32, 32, 3),
            original_test_size=10_000,
        ),
        DatasetConfig(
            key="svhn",
            display_name="SVHN",
            class_names=_DIGITS,
            train_class_counts=CIFAR_LIKE_COUNTS,
            image_shape=(32, 32, 3),
            original_test_size=26_032,
        ),
        DatasetConfig(
            key="celeba",
            display_name="CelebA",
            class_names=CELEBA_HAIR_ATTRIBUTES,
            train_class_counts=CELEBA_COUNTS,
            image_shape=(CELEBA_IMAGE_SIZE, CELEBA_IMAGE_SIZE, 3),
            original_test_size=None,
        ),
    )
}
