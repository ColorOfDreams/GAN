"""Loader cho các dataset gốc (không chỉnh sửa).

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import CELEBA_HAIR_ATTRIBUTES, CELEBA_IMAGE_SIZE, RAW_DIR

logger = logging.getLogger(__name__)

IGNORE_LABEL = -1  # ảnh CelebA không thuộc đúng một class màu tóc


@dataclass(frozen=True)
class Split:
    """Nhãn của cả split kèm một hàm lấy ảnh theo index.

    Ảnh được lấy lazily để CelebA không phải decode ~160k ảnh JPEG chỉ để giữ
    lại vài nghìn ảnh.
    """

    labels: np.ndarray
    fetch: Callable[[np.ndarray], np.ndarray]

    def __len__(self) -> int:
        return len(self.labels)

    def all_images(self) -> np.ndarray:
        return self.fetch(np.arange(len(self)))


@dataclass(frozen=True)
class RawDataset:
    train: Split
    test: Split


def _in_memory_split(images: np.ndarray, labels: np.ndarray) -> Split:
    images = np.ascontiguousarray(images, dtype=np.uint8)
    images.setflags(write=False)  # chặn việc vô tình sửa dữ liệu gốc
    labels = np.asarray(labels, dtype=np.int64)
    labels.setflags(write=False)
    return Split(labels=labels, fetch=lambda idx: images[idx])


def _torchvision_datasets():
    try:
        from torchvision import datasets
    except ImportError as exc:  # pragma: no cover - lỗi môi trường
        raise ImportError(
            "Cần torchvision để tải/đọc dataset gốc: pip install -r requirements.txt"
        ) from exc
    return datasets


def _load_mnist_like(dataset_cls, root: Path, download: bool) -> RawDataset:
    splits = []
    for train in (True, False):
        ds = dataset_cls(root=str(root), train=train, download=download)
        images = ds.data.numpy()[..., np.newaxis]  # (N, 28, 28) -> (N, 28, 28, 1)
        splits.append(_in_memory_split(images, ds.targets.numpy()))
    return RawDataset(*splits)


def load_mnist(root: Path, download: bool = True) -> RawDataset:
    return _load_mnist_like(_torchvision_datasets().MNIST, root, download)


def load_fashion_mnist(root: Path, download: bool = True) -> RawDataset:
    return _load_mnist_like(_torchvision_datasets().FashionMNIST, root, download)


def load_cifar10(root: Path, download: bool = True) -> RawDataset:
    datasets = _torchvision_datasets()
    splits = []
    for train in (True, False):
        ds = datasets.CIFAR10(root=str(root), train=train, download=download)
        splits.append(_in_memory_split(ds.data, np.array(ds.targets)))  # data đã là NHWC
    return RawDataset(*splits)


def load_svhn(root: Path, download: bool = True) -> RawDataset:
    """SVHN core train/test (73.257 / 26.032). torchvision đổi chữ số '10' -> nhãn 0."""
    datasets = _torchvision_datasets()
    splits = []
    for split in ("train", "test"):
        ds = datasets.SVHN(root=str(root), split=split, download=download)
        splits.append(_in_memory_split(ds.data.transpose(0, 2, 3, 1), ds.labels))  # NCHW -> NHWC
    return RawDataset(*splits)


def load_celeba(root: Path, download: bool = True) -> RawDataset:
    """CelebA với 5 class màu tóc mà DeepSMOTE sử dụng.

    Giả định (paper chỉ ghi "5 classes ... resized to 3 x 32 x 32"):
      * nhãn = vị trí trong CELEBA_HAIR_ATTRIBUTES; ảnh không có hoặc có nhiều
        hơn một thuộc tính trong số này là mơ hồ, nhận IGNORE_LABEL (không bao
        giờ được sample);
      * train = partition train chính thức của CelebA, test = partition test
        chính thức chỉ giữ các ảnh có nhãn (giữ toàn bộ, không downsample);
      * ảnh aligned 178x218 được center-crop về 178x178 rồi resize về 32x32
        bằng bộ lọc Lanczos (không làm méo tỉ lệ khung hình).
    """
    datasets = _torchvision_datasets()
    splits = []
    for split in ("train", "test"):
        try:
            ds = datasets.CelebA(root=str(root), split=split, target_type="attr", download=download)
        except Exception as exc:
            raise RuntimeError(
                "Không load được CelebA. torchvision tải CelebA từ Google Drive, nơi thường bị giới "
                "hạn lượt tải. Hãy tải thủ công img_align_celeba.zip, list_attr_celeba.txt, "
                "list_eval_partition.txt, identity_CelebA.txt, list_bbox_celeba.txt và "
                f"list_landmarks_align_celeba.txt vào {root / 'celeba'} (giải nén ảnh tại đó) "
                "rồi chạy lại."
            ) from exc
        labels = _celeba_hair_labels(ds)
        if split == "test":
            keep = np.flatnonzero(labels != IGNORE_LABEL)
            splits.append(Split(labels=labels[keep], fetch=_celeba_fetcher(ds, keep)))
        else:
            splits.append(Split(labels=labels, fetch=_celeba_fetcher(ds, np.arange(len(labels)))))
    return RawDataset(*splits)


def _celeba_hair_labels(ds) -> np.ndarray:
    columns = [ds.attr_names.index(name) for name in CELEBA_HAIR_ATTRIBUTES]
    hair = ds.attr[:, columns].numpy().astype(bool)  # torchvision lưu thuộc tính dạng {0, 1}
    labels = np.full(len(hair), IGNORE_LABEL, dtype=np.int64)
    exactly_one = hair.sum(axis=1) == 1
    labels[exactly_one] = hair[exactly_one].argmax(axis=1)
    return labels


def _celeba_fetcher(ds, positions: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    def fetch(idx: np.ndarray) -> np.ndarray:
        out = np.empty((len(idx), CELEBA_IMAGE_SIZE, CELEBA_IMAGE_SIZE, 3), dtype=np.uint8)
        for i, j in enumerate(np.asarray(idx)):
            image, _ = ds[int(positions[j])]
            out[i] = _center_crop_resize(image, CELEBA_IMAGE_SIZE)
        return out

    return fetch


def _center_crop_resize(image, size: int) -> np.ndarray:
    from PIL import Image

    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return np.asarray(image.resize((size, size), Image.LANCZOS), dtype=np.uint8)


LOADERS: dict[str, Callable[[Path, bool], RawDataset]] = {
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "cifar10": load_cifar10,
    "svhn": load_svhn,
    "celeba": load_celeba,
}


def load_raw_dataset(key: str, raw_dir: Path = RAW_DIR, download: bool = True) -> RawDataset:
    root = raw_dir / key
    root.mkdir(parents=True, exist_ok=True)
    logger.info("Đang load dataset gốc %s từ %s", key, root)
    return LOADERS[key](root, download)
