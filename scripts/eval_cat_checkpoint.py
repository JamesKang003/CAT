import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from src.cat.config import CATConfig
from src.cat.model import CATModel


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
META_PATH = PROCESSED_DIR / "meta.pkl"

TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"
TEST_BIN = PROCESSED_DIR / "test.bin"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_token_arrays():
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    test_data = np.memmap(TEST_BIN, dtype=np.uint16, mode="r")
    return train_data, val_data, test_data


def get_batch(split, train_data, val_data, test_data, block_size, batch_size, device):
    data = {"train": train_data, "val": val_data, "test": test_data}[split]

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack(
        [
            torch.from_numpy(np.array(data[i : i + block_size], dtype=np.int64))
            for i in ix
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(np.array(data[i + 1 : i + 1 + block_size], dtype=np.int64))
            for i in ix
        ]
    )

    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(
    model, train_data, val_data, test_data, block_size, batch_size, eval_iters, device
):
    model.eval()
    amp_enabled = device.type == "cuda"

    results = {}
    for split in ["val", "test"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(
                split, train_data, val_data, test_data, block_size, batch_size, device
            )
            with autocast(device_type="cuda", enabled=amp_enabled):
                _, loss = model(x, y)
            losses[i] = loss.item()

        ce = losses.mean().item()
        ppl = math.exp(ce) if ce < 20 else float("inf")
        results[f"{split}_ce"] = ce
        results[f"{split}_ppl"] = ppl

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path")
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    meta = load_meta()
    train_data, val_data, test_data = load_token_arrays()

    ckpt_path = Path(args.ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)

    config = CATConfig(**checkpoint["config"])
    model = CATModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    metrics = evaluate(
        model=model,
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        block_size=config.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        device=device,
    )

    print(f"\nCheckpoint: {ckpt_path}")
    print(f"Step: {checkpoint.get('step', 'N/A')}")
    print(f"val_ce   : {metrics['val_ce']:.4f}")
    print(f"val_ppl  : {metrics['val_ppl']:.2f}")
    print(f"test_ce  : {metrics['test_ce']:.4f}")
    print(f"test_ppl : {metrics['test_ppl']:.2f}")


if __name__ == "__main__":
    main()
