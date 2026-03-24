import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from src.cat.config import CATConfig
from src.cat.model import CATModel


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHECKPOINT_DIR = BASE_DIR / "models" / "checkpoints"
META_PATH = PROCESSED_DIR / "meta.pkl"

TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"
TEST_BIN = PROCESSED_DIR / "test.bin"

RESUME_CKPT = CHECKPOINT_DIR / "ckpt.pt"
FINAL_CKPT = CHECKPOINT_DIR / "cat_resumed_final.pt"

# 추가 학습용 세팅
batch_size = 16
block_size = 256
resume_iters = 2500
eval_interval = 250
eval_iters = 50

# 원래보다 낮은 lr로 미세 조정
learning_rate = 1.5e-4
min_lr = 3e-5
warmup_iters = 200

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
grad_accum_steps = 4

seed = 42


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def load_token_arrays():
    train_data = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_data = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    test_data = np.memmap(TEST_BIN, dtype=np.uint16, mode="r")
    return train_data, val_data, test_data


def get_batch(split, train_data, val_data, test_data, device):
    data = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }[split]

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

    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, test_data, device, amp_enabled):
    model.eval()
    out = {}

    for split in ["val", "test"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, test_data, device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                _, loss = model(x, y)
            losses[k] = loss.item()

        mean_loss = losses.mean().item()
        ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
        out[f"{split}_loss"] = mean_loss
        out[f"{split}_ppl"] = ppl

    model.train()
    return out


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters

    if it > resume_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (resume_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(path, model, optimizer, step, best_val_loss, config):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config.__dict__,
    }
    torch.save(checkpoint, path)


def main():
    set_seed(seed)
    device = get_device()
    amp_enabled = device.type == "cuda"

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not RESUME_CKPT.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {RESUME_CKPT}")

    meta = load_meta()
    train_data, val_data, test_data = load_token_arrays()

    checkpoint = torch.load(RESUME_CKPT, map_location=device)
    config = CATConfig(**checkpoint["config"])

    model = CATModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    # optimizer state는 굳이 이어받아도 되고 안 받아도 되는데,
    # 지금은 "낮은 lr로 추가 미세조정" 목적이라 새 optimizer가 더 깔끔함.

    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    start_step = checkpoint.get("step", 0)

    print(f"Resuming from: {RESUME_CKPT}")
    print(f"Original checkpoint step: {start_step}")
    print(f"Best val loss so far: {best_val_loss:.4f}")

    t0 = time.time()
    model.train()

    for step in range(resume_iters + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % eval_interval == 0:
            metrics = estimate_loss(
                model, train_data, val_data, test_data, device, amp_enabled
            )
            elapsed = time.time() - t0

            print(
                f"resume step {step:5d} | "
                f"lr {lr:.6f} | "
                f"val loss {metrics['val_loss']:.4f} | "
                f"val ppl {metrics['val_ppl']:.2f} | "
                f"test loss {metrics['test_loss']:.4f} | "
                f"test ppl {metrics['test_ppl']:.2f} | "
                f"time {elapsed:.1f}s"
            )

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                save_checkpoint(
                    RESUME_CKPT,
                    model,
                    optimizer,
                    start_step + step,
                    best_val_loss,
                    config,
                )
                print(f"Updated best checkpoint: {RESUME_CKPT}")

        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum_steps):
            x, y = get_batch("train", train_data, val_data, test_data, device)
            with autocast(device_type="cuda", enabled=amp_enabled):
                _, loss = model(x, y)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

    save_checkpoint(
        FINAL_CKPT, model, optimizer, start_step + resume_iters, best_val_loss, config
    )
    print(f"Saved resumed final checkpoint to {FINAL_CKPT}")


if __name__ == "__main__":
    main()
