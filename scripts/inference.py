import pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.cat.config import CATConfig
from src.cat.model import CATModel


BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"
META_PATH = BASE_DIR / "data" / "processed" / "meta.pkl"

# 보통 best checkpoint부터 보는 게 맞다.
CKPT_PATH = BASE_DIR / "models" / "checkpoints" / "ckpt.pt"
# 필요하면 final로 바꿔도 됨:
# CKPT_PATH = BASE_DIR / "models" / "checkpoints" / "cat_final.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


def top_k_filter(logits: torch.Tensor, k: int | None):
    if k is None:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = float("-inf")
    return logits


@torch.no_grad()
def generate(
    model: CATModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    block_size: int,
    temperature: float = 1.0,
    top_k: int | None = 50,
    eos_id: int | None = None,
):
    model.eval()

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        logits = top_k_filter(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, next_id), dim=1)

        if eos_id is not None and next_id.item() == eos_id:
            break

    return idx


def decode_generated(
    tokenizer: Tokenizer, ids: list[int], bos_id: int, eos_id: int
) -> str:
    filtered = [i for i in ids if i not in (bos_id, eos_id)]
    text = tokenizer.decode(filtered)
    return text.strip()


def main():
    device = get_device()
    print(f"Using device: {device}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    meta = load_meta()

    vocab_size = meta["vocab_size"]
    bos_id = meta["bos_id"]
    eos_id = meta["eos_id"]

    checkpoint = torch.load(CKPT_PATH, map_location=device)

    config_dict = checkpoint["config"]
    config = CATConfig(**config_dict)

    model = CATModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {CKPT_PATH}")
    print(f"Vocab size: {vocab_size}")
    print(f"Block size: {config.block_size}")

    prompts = [
        "The patient presented with progressive shortness of breath and",
        "Results showed a significant increase in inflammatory markers after",
        "Computed tomography of the chest revealed",
        "In this study, patients with hypertension were treated with",
        "The findings suggest that early intervention may improve",
    ]

    max_new_tokens = 120
    temperature = 0.8
    top_k = 50

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
            eos_id=eos_id,
        )

        generated_ids = y[0].tolist()
        decoded = decode_generated(tokenizer, generated_ids, bos_id, eos_id)

        print("\n[Generated]")
        print(decoded)

    print("\nDone.")


if __name__ == "__main__":
    main()
