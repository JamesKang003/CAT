import argparse
import pickle
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.cat.config import CATConfig
from src.cat.model import CATModel


BASE_DIR = Path(__file__).resolve().parent.parent
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"
META_PATH = BASE_DIR / "data" / "processed" / "meta.pkl"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_meta():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)


@torch.no_grad()
def generate_cat(model, tokenizer, bos_id, eos_id, prompt, max_new_tokens, device):
    prompt_ids = tokenizer.encode(prompt).ids
    x = torch.tensor([[bos_id] + prompt_ids], dtype=torch.long, device=device)

    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = x[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        probs = torch.softmax(logits / 0.7, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)
        if next_id.item() == eos_id:
            break

    ids = [i for i in x[0].tolist() if i not in (bos_id, eos_id)]
    return tokenizer.decode(ids, skip_special_tokens=True)


@torch.no_grad()
def generate_gpt2(model, tokenizer, prompt, max_new_tokens, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = get_device()
    meta = load_meta()

    # CAT
    cat_tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    checkpoint = torch.load(args.ckpt, map_location=device)
    cat_config = CATConfig(**checkpoint["config"])
    cat_model = CATModel(cat_config)
    cat_model.load_state_dict(checkpoint["model_state_dict"])
    cat_model.to(device)

    # GPT-2 small
    gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    prompts = [
        "The patient presented with progressive shortness of breath and",
        "Results showed a significant increase in inflammatory markers after",
        "Computed tomography of the chest revealed",
        "In this study, patients with hypertension were treated with",
        "The findings suggest that early intervention may improve",
    ]

    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 100)
        print(f"[Prompt {i}] {prompt}")

        cat_out = generate_cat(
            model=cat_model,
            tokenizer=cat_tokenizer,
            bos_id=meta["bos_id"],
            eos_id=meta["eos_id"],
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print("\n[CAT]")
        print(cat_out)

        gpt2_out = generate_gpt2(
            model=gpt2_model,
            tokenizer=gpt2_tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        print("\n[GPT-2 small]")
        print(gpt2_out)


if __name__ == "__main__":
    main()
