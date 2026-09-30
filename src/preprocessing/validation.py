"""Kiểm tra dataset mất cân bằng đã tạo có tuân theo protocol DeepSMOTE hay không."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import DatasetConfig
from .loaders import RawDataset
from .sampling import sample_imbalanced_indices


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_dataset(
    cfg: DatasetConfig,
    raw: RawDataset,
    seed: int,
    train_x: np.ndarray,
    train_y: np.ndarray,
    source_indices: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> list[CheckResult]:
    """Chạy toàn bộ các kiểm tra trên dataset *đã được load lại từ đĩa*."""
    return [
        _check_class_counts(cfg, train_y),
        _check_total(cfg, train_x, train_y, source_indices),
        _check_no_duplicates(cfg, train_y, source_indices),
        _check_labels_match_original(raw, train_y, source_indices),
        _check_images_are_original(cfg, raw, train_x, source_indices),
        _check_reproducible(cfg, raw, seed, source_indices),
        _check_test_unchanged(cfg, raw, test_x, test_y),
    ]


def _check_class_counts(cfg: DatasetConfig, train_y: np.ndarray) -> CheckResult:
    in_range = bool(np.all((train_y >= 0) & (train_y < cfg.num_classes)))
    counts = np.bincount(train_y[(train_y >= 0) & (train_y < cfg.num_classes)], minlength=cfg.num_classes)
    expected = np.array(cfg.train_class_counts)
    mismatches = {c: (int(counts[c]), int(expected[c])) for c in range(cfg.num_classes) if counts[c] != expected[c]}
    passed = in_range and not mismatches
    detail = (
        "số mẫu mỗi class khớp cấu hình" if passed
        else f"nhãn hợp lệ={in_range}, (thực tế, mong đợi)={mismatches}"
    )
    return CheckResult("class_distribution", passed, detail)


def _check_total(cfg, train_x, train_y, source_indices) -> CheckResult:
    sizes = {len(train_x), len(train_y), len(source_indices)}
    shape_ok = train_x.shape[1:] == cfg.image_shape and train_x.dtype == np.uint8
    passed = sizes == {cfg.total_train} and shape_ok
    detail = f"tổng={cfg.total_train}, kích thước ảnh={train_x.shape[1:]}, dtype={train_x.dtype}"
    return CheckResult("total_samples", passed, detail if passed else f"số lượng={sizes}; {detail}")


def _check_no_duplicates(cfg, train_y, source_indices) -> CheckResult:
    dupes = {
        c: int(np.sum(train_y == c) - len(np.unique(source_indices[train_y == c])))
        for c in range(cfg.num_classes)
    }
    dupes = {c: n for c, n in dupes.items() if n}
    overall_unique = len(np.unique(source_indices)) == len(source_indices)
    passed = overall_unique and not dupes
    return CheckResult(
        "no_duplicate_indices", passed,
        "mọi source index đều duy nhất" if passed else f"trùng lặp theo class: {dupes}",
    )


def _check_labels_match_original(raw, train_y, source_indices) -> CheckResult:
    passed = bool(np.array_equal(raw.train.labels[source_indices], train_y))
    return CheckResult(
        "labels_match_original", passed,
        "nhãn đã lưu == nhãn gốc tại source index" if passed else "nhãn không khớp",
    )


def _check_images_are_original(cfg, raw, train_x, source_indices) -> CheckResult:
    passed = bool(np.array_equal(raw.train.fetch(source_indices), train_x))
    return CheckResult(
        "images_are_original", passed,
        "ảnh đã lưu == ảnh gốc (không có mẫu tổng hợp/bị sửa)" if passed else "ảnh không khớp",
    )


def _check_reproducible(cfg, raw, seed, source_indices) -> CheckResult:
    again = sample_imbalanced_indices(raw.train.labels, cfg.train_class_counts, seed)
    other = sample_imbalanced_indices(raw.train.labels, cfg.train_class_counts, seed + 1)
    same = bool(np.array_equal(again, source_indices))
    seed_matters = not np.array_equal(other, source_indices)
    passed = same and seed_matters
    detail = f"sample lại với seed {seed} cho index giống hệt; seed {seed + 1} cho kết quả khác"
    return CheckResult(
        "seed_reproducible", passed,
        detail if passed else f"giống nhau={same}, seed có tác dụng={seed_matters}",
    )


def _check_test_unchanged(cfg, raw, test_x, test_y) -> CheckResult:
    size_ok = cfg.original_test_size is None or len(test_y) == cfg.original_test_size
    passed = (
        size_ok
        and bool(np.array_equal(raw.test.labels, test_y))
        and bool(np.array_equal(raw.test.all_images(), test_x))
    )
    detail = f"test set giống hệt bản gốc ({len(test_y)} mẫu, không resample)"
    return CheckResult(
        "test_set_unchanged", passed,
        detail if passed else f"kích thước đúng={size_ok}; nội dung khác bản gốc",
    )
