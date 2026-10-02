"""Sinh ảnh synthetic cho các class thiểu số (DeepSMOTE tầng 2) và ghép với ảnh thật.

Dùng encoder/decoder đã train ở `train_autoencoder.py` (KHÔNG train thêm ở đây).
Với mỗi class có số mẫu ít hơn class đông nhất: encode ảnh thật của class đó ->
SMOTE trong latent space (kNN + nội suy tuyến tính, xem latent_smote.py) -> decode
-> ảnh synthetic. Ghép ảnh synthetic với TOÀN BỘ ảnh thật gốc (không bớt, không
sửa) -> dataset đã balance.

Cách dùng:
    python src/deepsmote/generate_balanced_dataset.py --dataset mnist --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):  # chạy trực tiếp dạng script: cho phép import `preprocessing`/`deepsmote`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepsmote.config import DEFAULT_TRAIN_CONFIG, TrainConfig  # noqa: E402
from deepsmote.data import denormalize_to_uint8, load_train_tensors  # noqa: E402
from deepsmote.latent_smote import smote_in_latent_space  # noqa: E402
from deepsmote.models import Decoder, Encoder  # noqa: E402
from deepsmote.train_autoencoder import MODEL_DIRNAME, TRAIN_CONFIG_FILE  # noqa: E402
from preprocessing.config import DATASET_CONFIGS, DEFAULT_SEED, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger("generate_balanced_dataset")

BALANCED_FILE = "balanced_train.npz"
BALANCED_METADATA_FILE = "balanced_metadata.json"


def generate_balanced_dataset(
    dataset_key: str,
    seed: int = DEFAULT_SEED,
    processed_dir: Path = PROCESSED_DIR,
    k_neighbors: int = DEFAULT_TRAIN_CONFIG.k_neighbors,
    device: str | None = None,
) -> tuple[Path, dict]:
    """Sinh dataset balance, lưu `balanced_train.npz` + metadata. Trả về (đường dẫn, metadata)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, dcfg, out_dir = load_train_tensors(dataset_key, seed, processed_dir)
    model_dir = out_dir / MODEL_DIRNAME

    train_cfg_path = model_dir / TRAIN_CONFIG_FILE
    enc_path, dec_path = model_dir / "encoder_best.pth", model_dir / "decoder_best.pth"
    if not (train_cfg_path.exists() and enc_path.exists()):
        raise FileNotFoundError(
            f"Chưa có encoder/decoder đã train tại {model_dir}. Hãy chạy trước:\n"
            f"  python src/deepsmote/train_autoencoder.py --dataset {dataset_key} --seed {seed}"
        )
    trained_cfg = json.loads(train_cfg_path.read_text(encoding="utf-8"))
    dim_h, n_z = trained_cfg["dim_h"], trained_cfg["n_z"]  # phải khớp model đã lưu, không lấy từ CLI

    encoder = Encoder(dcfg.image_shape[2], dim_h, n_z).to(device)
    decoder = Decoder(dcfg.image_shape[2], dcfg.image_shape[0], dim_h, n_z).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    decoder.load_state_dict(torch.load(dec_path, map_location=device))
    encoder.eval()
    decoder.eval()

    images_dev = images.to(device)
    target = max(dcfg.train_class_counts)

    synthetic_images, synthetic_labels = [], []
    per_class_report: dict[str, dict] = {}
    with torch.no_grad():
        for c, count in enumerate(dcfg.train_class_counts):
            n_needed = target - count
            per_class_report[str(c)] = {"real": count, "synthetic": n_needed, "total": target}
            if n_needed <= 0:
                continue

            class_idx = (labels == c).nonzero(as_tuple=True)[0]
            latent = encoder(images_dev[class_idx]).cpu().numpy()

            synth_latent = smote_in_latent_space(latent, n_needed, k_neighbors, seed=seed + c)
            decoded = decoder(torch.from_numpy(synth_latent).to(device)).cpu()

            synthetic_images.append(denormalize_to_uint8(decoded))
            synthetic_labels.append(np.full(n_needed, c, dtype=np.int64))
            logger.info("Class %d: %d thật + %d synthetic = %d", c, count, n_needed, target)

    real_images_np = denormalize_to_uint8(images)
    real_labels_np = labels.numpy()

    if synthetic_images:
        synth_x = np.concatenate(synthetic_images, axis=0)
        synth_y = np.concatenate(synthetic_labels, axis=0)
        balanced_x = np.concatenate([real_images_np, synth_x], axis=0)
        balanced_y = np.concatenate([real_labels_np, synth_y], axis=0)
        is_synthetic = np.concatenate(
            [np.zeros(len(real_labels_np), dtype=bool), np.ones(len(synth_y), dtype=bool)]
        )
    else:
        balanced_x, balanced_y = real_images_np, real_labels_np
        is_synthetic = np.zeros(len(real_labels_np), dtype=bool)

    np.savez_compressed(model_dir / BALANCED_FILE, x=balanced_x, y=balanced_y, is_synthetic=is_synthetic)

    metadata = {
        "dataset": dcfg.display_name,
        "dataset_key": dataset_key,
        "seed": seed,
        "method": "DeepSMOTE (autoencoder + SMOTE trong latent space)",
        "target_per_class": target,
        "per_class": per_class_report,
        "total_real": int(len(real_labels_np)),
        "total_synthetic": int(is_synthetic.sum()),
        "total_balanced": int(len(balanced_y)),
        "train_config": trained_cfg,
        "k_neighbors": k_neighbors,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (model_dir / BALANCED_METADATA_FILE).write_text(
        json.dumps(metadata, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    out_path = model_dir / BALANCED_FILE
    logger.info("Đã lưu dataset balance: %s (%d thật + %d synthetic = %d)", out_path,
                metadata["total_real"], metadata["total_synthetic"], metadata["total_balanced"])
    return out_path, metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--k-neighbors", type=int, default=DEFAULT_TRAIN_CONFIG.k_neighbors)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args(argv)
    generate_balanced_dataset(args.dataset, seed=args.seed, processed_dir=args.processed_dir,
                              k_neighbors=args.k_neighbors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
