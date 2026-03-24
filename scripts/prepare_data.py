import pickle
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


# ---------------------------------------------------------------------
# Project: Clinically-Aware Transformer (CAT)
# Module: Tokenization + binary dataset serialization
# Purpose:
#   - Read train/val/test raw text files
#   - Encode each article with tokenizer.json
#   - Add BOS/EOS tokens
#   - Save as uint16 token streams (.bin)
#   - Save metadata needed for training/inference
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INTERIM_DIR = BASE_DIR / "data" / "interim"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TOKENIZER_PATH = BASE_DIR / "models" / "tokenizer.json"

TRAIN_FILE = INTERIM_DIR / "train_raw.txt"
VAL_FILE = INTERIM_DIR / "val_raw.txt"
TEST_FILE = INTERIM_DIR / "test_raw.txt"


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")

    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def encode_split(
    lines: list[str], tokenizer: Tokenizer, bos_id: int, eos_id: int
) -> np.ndarray:
    token_ids: list[int] = []

    for i, line in enumerate(lines):
        ids = tokenizer.encode(line).ids
        token_ids.extend([bos_id] + ids + [eos_id])

        if (i + 1) % 1000 == 0:
            print(f"Encoded {i + 1}/{len(lines)} documents")

    arr = np.array(token_ids, dtype=np.uint16)
    return arr


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

    vocab_size = tokenizer.get_vocab_size()
    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    pad_id = tokenizer.token_to_id("[PAD]")
    unk_id = tokenizer.token_to_id("[UNK]")

    if None in (bos_id, eos_id, pad_id, unk_id):
        raise ValueError("Tokenizer is missing one or more required special tokens.")

    print(f"Loaded tokenizer from {TOKENIZER_PATH}")
    print(f"Vocab size: {vocab_size}")
    print(f"[BOS]={bos_id}, [EOS]={eos_id}, [PAD]={pad_id}, [UNK]={unk_id}")

    train_lines = load_lines(TRAIN_FILE)
    val_lines = load_lines(VAL_FILE)
    test_lines = load_lines(TEST_FILE)

    print(f"Train docs: {len(train_lines)}")
    print(f"Val docs  : {len(val_lines)}")
    print(f"Test docs : {len(test_lines)}")

    train_ids = encode_split(train_lines, tokenizer, bos_id, eos_id)
    val_ids = encode_split(val_lines, tokenizer, bos_id, eos_id)
    test_ids = encode_split(test_lines, tokenizer, bos_id, eos_id)

    train_bin = PROCESSED_DIR / "train.bin"
    val_bin = PROCESSED_DIR / "val.bin"
    test_bin = PROCESSED_DIR / "test.bin"
    meta_pkl = PROCESSED_DIR / "meta.pkl"

    train_ids.tofile(train_bin)
    val_ids.tofile(val_bin)
    test_ids.tofile(test_bin)

    meta = {
        "vocab_size": vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "unk_id": unk_id,
        "train_docs": len(train_lines),
        "val_docs": len(val_lines),
        "test_docs": len(test_lines),
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "test_tokens": len(test_ids),
        "dtype": "uint16",
    }

    with meta_pkl.open("wb") as f:
        pickle.dump(meta, f)

    print("\n--- Data Preparation Complete ---")
    print(f"Saved: {train_bin}")
    print(f"Saved: {val_bin}")
    print(f"Saved: {test_bin}")
    print(f"Saved: {meta_pkl}")
    print(f"Train tokens: {len(train_ids):,}")
    print(f"Val tokens  : {len(val_ids):,}")
    print(f"Test tokens : {len(test_ids):,}")


if __name__ == "__main__":
    main()
