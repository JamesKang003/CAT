import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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

batch_size = 8
block_size = 512
max_iters = 6000
eval_interval = 500
eval_iters = 20  # 50 -> 20으로 줄여서 eval 속도 개선

learning_rate = 4e-4
min_lr = 4e-5
warmup_iters = 500

weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
grad_accum_steps = 8

seed = 42

# -----------------------------
# Lightweight unlikelihood settings
# -----------------------------
ul_weight = 0.15
ul_window = 16
ignore_token_ids = {0, 1, 2, 3}  # [UNK], [PAD], [BOS], [EOS]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def get_batch(split, train_data, val_data, test_data, device):
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


def sequence_cross_entropy_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    B, T, V = logits.shape
    return F.cross_entropy(
        logits.view(B * T, V),
        targets.view(B * T),
    )


def repetition_unlikelihood_loss_light(
    logits: torch.Tensor,
    targets: torch.Tensor,
    window: int,
    ignore_ids: set[int],
) -> torch.Tensor:
    """
    Lightweight repetition unlikelihood:
    For each position, collect recently seen target tokens (excluding current gold token),
    then penalize high probability assigned to those repeated candidates.

    Avoids building a full [B, T, V] candidate mask.
    """
    B, T, V = logits.shape
    probs = F.softmax(logits, dim=-1)

    losses = []
    eps = 1e-6

    # Python loops remain, but complexity is much smaller than full vocab mask.
    for b in range(B):
        for t in range(T):
            start = max(0, t - window)
            history = targets[b, start:t].tolist()
            if not history:
                continue

            current_target = targets[b, t].item()
            seen = set(history)

            if current_target in seen:
                seen.remove(current_target)

            seen = [tok for tok in seen if tok not in ignore_ids]
            if not seen:
                continue

            candidate_ids = torch.tensor(seen, device=logits.device, dtype=torch.long)
            candidate_probs = probs[b, t, candidate_ids]

            loss_t = -torch.log(torch.clamp(1.0 - candidate_probs, min=eps)).mean()
            losses.append(loss_t)

    if not losses:
        return torch.tensor(0.0, device=logits.device)

    return torch.stack(losses).mean()


def compute_total_loss(
    model: CATModel,
    x: torch.Tensor,
    y: torch.Tensor,
    ul_window: int,
    ul_weight: float,
    ignore_ids: set[int],
):
    logits, _ = model(x, y)

    ce_loss = sequence_cross_entropy_loss(logits, y)
    ul_loss = repetition_unlikelihood_loss_light(
        logits=logits,
        targets=y,
        window=ul_window,
        ignore_ids=ignore_ids,
    )

    total_loss = ce_loss + ul_weight * ul_loss
    return logits, total_loss, ce_loss.detach(), ul_loss.detach()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, test_data, device, amp_enabled):
    model.eval()
    out = {}

    for split in ["val", "test"]:
        total_losses = torch.zeros(eval_iters)
        ce_losses = torch.zeros(eval_iters)
        ul_losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = get_batch(split, train_data, val_data, test_data, device)

            with autocast(device_type="cuda", enabled=amp_enabled):
                _, total_loss, ce_loss, ul_loss = compute_total_loss(
                    model=model,
                    x=x,
                    y=y,
                    ul_window=ul_window,
                    ul_weight=ul_weight,
                    ignore_ids=ignore_token_ids,
                )

            total_losses[k] = total_loss.item()
            ce_losses[k] = ce_loss.item()
            ul_losses[k] = ul_loss.item()

        mean_total = total_losses.mean().item()
        mean_ce = ce_losses.mean().item()
        mean_ul = ul_losses.mean().item()
        ppl = math.exp(mean_ce) if mean_ce < 20 else float("inf")

        out[f"{split}_loss"] = mean_total
        out[f"{split}_ce"] = mean_ce
        out[f"{split}_ul"] = mean_ul
        out[f"{split}_ppl"] = ppl

    model.train()
    return out


def get_lr(it: int) -> float:
    if it < warmup_iters:
        return learning_rate * (it + 1) / warmup_iters
    if it > max_iters:
        return min_lr

    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def save_checkpoint(path, model, optimizer, step, best_val_ce, config):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_val_ce": best_val_ce,
        "config": config.__dict__,
    }
    torch.save(checkpoint, path)


def main():
    set_seed(seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()
    amp_enabled = device.type == "cuda"

    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    meta = load_meta()
    print("Loaded meta:", meta)

    train_data, val_data, test_data = load_token_arrays()

    config = CATConfig(
        block_size=block_size,
        vocab_size=meta["vocab_size"],
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.2,
        bias=False,
    )

    model = CATModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=1e-8,
        weight_decay=weight_decay,
    )

    scaler = GradScaler("cuda", enabled=amp_enabled)

    best_val_ce = float("inf")
    best_ckpt_path = CHECKPOINT_DIR / "ckpt_v3.pt"
    final_ckpt_path = CHECKPOINT_DIR / "cat_v3_final.pt"

    model.train()
    t0 = time.time()

    for step in range(max_iters + 1):
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if step % eval_interval == 0:
            metrics = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                device=device,
                amp_enabled=amp_enabled,
            )
            elapsed = time.time() - t0

            print(
                f"step {step:5d} | "
                f"lr {lr:.6f} | "
                f"val total {metrics['val_loss']:.4f} | "
                f"val ce {metrics['val_ce']:.4f} | "
                f"val ul {metrics['val_ul']:.4f} | "
                f"val ppl {metrics['val_ppl']:.2f} | "
                f"test ce {metrics['test_ce']:.4f} | "
                f"test ul {metrics['test_ul']:.4f} | "
                f"test ppl {metrics['test_ppl']:.2f} | "
                f"time {elapsed:.1f}s"
            )

            if metrics["val_ce"] < best_val_ce:
                best_val_ce = metrics["val_ce"]
                save_checkpoint(
                    best_ckpt_path, model, optimizer, step, best_val_ce, config
                )
                print(f"Saved best checkpoint to {best_ckpt_path}")

        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum_steps):
            x, y = get_batch("train", train_data, val_data, test_data, device)

            with autocast(device_type="cuda", enabled=amp_enabled):
                _, total_loss, _, _ = compute_total_loss(
                    model=model,
                    x=x,
                    y=y,
                    ul_window=ul_window,
                    ul_weight=ul_weight,
                    ignore_ids=ignore_token_ids,
                )
                total_loss = total_loss / grad_accum_steps

            scaler.scale(total_loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

    save_checkpoint(final_ckpt_path, model, optimizer, max_iters, best_val_ce, config)
    print(f"Saved final checkpoint to {final_ckpt_path}")


if __name__ == "__main__":
    main()
