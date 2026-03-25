import pickle
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.cat.config import CATConfig
from src.cat.model import CATModel


BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"
META_PATH = BASE_DIR / "data" / "processed" / "meta.pkl"

# 기본은 best checkpoint
CKPT_PATH = BASE_DIR / "models" / "checkpoints" / "ckpt_v2.pt"
# 이어학습 final 보고 싶으면 이걸로 바꿔도 됨:
# CKPT_PATH = BASE_DIR / "models" / "checkpoints" / "cat_resumed_final.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def top_k_filter(logits: torch.Tensor, k: Optional[int]):
    if k is None:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0:
        return logits

    for batch_idx in range(logits.size(0)):
        seen_tokens = set(generated_ids[batch_idx].tolist())
        for token_id in seen_tokens:
            if logits[batch_idx, token_id] < 0:
                logits[batch_idx, token_id] *= penalty
            else:
                logits[batch_idx, token_id] /= penalty

    return logits


def clean_generated_text(text: str) -> str:
    text = text.replace("Ġ", " ")
    text = text.replace("Ċ", "\n")
    text = text.replace("âĢĵ", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def generate(
    model: CATModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 0.7,
    top_k: Optional[int] = 40,
    repetition_penalty: float = 1.15,
    eos_id: Optional[int] = None,
):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        logits = logits / temperature
        logits = apply_repetition_penalty(logits, idx, repetition_penalty)
        logits = top_k_filter(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    return idx


def decode_generated(
    tokenizer: Tokenizer,
    ids: list[int],
    bos_id: int,
    eos_id: int,
) -> str:
    filtered = [i for i in ids if i not in (bos_id, eos_id)]
    text = tokenizer.decode(filtered, skip_special_tokens=True)
    text = clean_generated_text(text)
    return text.strip()


def main():
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    meta = load_meta()

    bos_id = meta["bos_id"]
    eos_id = meta["eos_id"]

    checkpoint = torch.load(CKPT_PATH, map_location=device)
    config = CATConfig(**checkpoint["config"])

    model = CATModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {CKPT_PATH}")
    print(f"Vocab size: {meta['vocab_size']}")
    print(f"Block size: {config.block_size}")

    prompts = [
        "The patient presented with progressive shortness of breath and",
        "Results showed a significant increase in inflammatory markers after",
        "Computed tomography of the chest revealed",
        "In this study, patients with hypertension were treated with",
        "The findings suggest that early intervention may improve",
    ]

    max_new_tokens = 80
    temperature = 0.65
    top_k = 40
    repetition_penalty = 1.20

    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 100)
        print(f"[Prompt {i}]")
        print(prompt)

        prompt_ids = tokenizer.encode(prompt).ids
        input_ids = [bos_id] + prompt_ids
        x = torch.tensor([input_ids], dtype=torch.long, device=device)

        y = generate(
            model=model,
            idx=x,
            max_new_tokens=max_new_tokens,
            block_size=config.block_size,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_id=eos_id,
        )

        generated_ids = y[0].tolist()
        decoded = decode_generated(tokenizer, generated_ids, bos_id, eos_id)

        print("\n[Generated]")
        print(decoded)

    print("\nDone.")


if __name__ == "__main__":
    main()
