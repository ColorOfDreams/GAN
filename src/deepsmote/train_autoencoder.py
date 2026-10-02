"""Train autoencoder DeepSMOTE (tầng 2) trên dataset mất cân bằng đã tạo ở tầng 1.

Loss mỗi bước = reconstruction trên batch random (toàn bộ class, tỉ lệ tự nhiên
mất cân bằng) + reconstruction THIÊN LỆCH trên 100 mẫu của 1 class ngẫu nhiên
(biased_get_class trong code gốc) — mục đích là ép autoencoder học tốt cả những
class hiếm, vốn hiếm khi xuất hiện đủ trong 1 batch random. Phần này KHÔNG nội
suy latent — nội suy thật sự (SMOTE) chỉ diễn ra sau, ở generate_balanced_dataset.py.

Cách dùng:
    python src/deepsmote/train_autoencoder.py --dataset mnist --seed 42 --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if __package__ in (None, ""):  # chạy trực tiếp dạng script: cho phép import `preprocessing`/`deepsmote`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepsmote.config import DEFAULT_TRAIN_CONFIG, TrainConfig  # noqa: E402
from deepsmote.data import load_train_tensors  # noqa: E402
from deepsmote.models import Decoder, Encoder  # noqa: E402
from preprocessing.config import DATASET_CONFIGS, DEFAULT_SEED, PROCESSED_DIR  # noqa: E402

logger = logging.getLogger("train_autoencoder")

MODEL_DIRNAME = "deepsmote"
TRAIN_CONFIG_FILE = "train_config.json"


def train_autoencoder(
    dataset_key: str,
    seed: int = DEFAULT_SEED,
    processed_dir: Path = PROCESSED_DIR,
    cfg: TrainConfig = DEFAULT_TRAIN_CONFIG,
    device: str | None = None,
) -> Path:
    """Train encoder/decoder, lưu trọng số (best + final) và cấu hình đã dùng. Trả về thư mục model."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    images, labels, dcfg, out_dir = load_train_tensors(dataset_key, seed, processed_dir)
    images, labels = images.to(device), labels.to(device)

    encoder = Encoder(dcfg.image_shape[2], cfg.dim_h, cfg.n_z).to(device)
    decoder = Decoder(dcfg.image_shape[2], dcfg.image_shape[0], cfg.dim_h, cfg.n_z).to(device)
    criterion = nn.MSELoss()
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=cfg.lr)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=cfg.lr)

    loader = DataLoader(TensorDataset(images, labels), batch_size=cfg.batch_size, shuffle=True)

    model_dir = out_dir / MODEL_DIRNAME
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / TRAIN_CONFIG_FILE).write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    logger.info("Train trên %s | device=%s | %d mẫu | %d class", dcfg.display_name, device,
                len(labels), dcfg.num_classes)

    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        t0 = time.time()

        for batch_x, _ in loader:
            enc_opt.zero_grad()
            dec_opt.zero_grad()

            z = encoder(batch_x)
            x_hat = decoder(z)
            mse = criterion(x_hat, batch_x)

            # Reconstruction thiên lệch theo 1 class ngẫu nhiên — không phải nội suy, xem docstring module.
            tc = int(torch.randint(0, dcfg.num_classes, (1,)))
            class_idx = (labels == tc).nonzero(as_tuple=True)[0]
            if len(class_idx) > 0:
                n_pick = min(cfg.biased_samples_per_step, len(class_idx))
                pick = class_idx[torch.randperm(len(class_idx), device=device)[:n_pick]]
                xclass = images[pick]
                mse2 = criterion(decoder(encoder(xclass)), xclass)
            else:
                mse2 = torch.zeros((), device=device)

            loss = mse + mse2
            loss.backward()
            enc_opt.step()
            dec_opt.step()
            total_loss += loss.item() * batch_x.size(0)

        avg_loss = total_loss / len(labels)
        logger.info("Epoch %d/%d - loss=%.6f (%.1fs)", epoch + 1, cfg.epochs, avg_loss, time.time() - t0)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), model_dir / "encoder_best.pth")
            torch.save(decoder.state_dict(), model_dir / "decoder_best.pth")

    torch.save(encoder.state_dict(), model_dir / "encoder_final.pth")
    torch.save(decoder.state_dict(), model_dir / "decoder_final.pth")
    logger.info("Đã lưu encoder/decoder (best + final) vào %s", model_dir)
    return model_dir


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
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    args = parse_args(argv)
    cfg = TrainConfig(dim_h=args.dim_h, n_z=args.n_z, lr=args.lr, epochs=args.epochs,
                      batch_size=args.batch_size)
    train_autoencoder(args.dataset, seed=args.seed, processed_dir=args.processed_dir, cfg=cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
