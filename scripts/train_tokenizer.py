from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = BASE_DIR / "data" / "interim" / "train_raw.txt"
TOKENIZER_OUT = BASE_DIR / "models" / "tokenizer.json"


def line_iterator(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def main():
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"{TRAIN_FILE} not found")

    print(f"Training tokenizer on: {TRAIN_FILE}")

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        min_frequency=2,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=True,
    )

    tokenizer.train_from_iterator(line_iterator(TRAIN_FILE), trainer=trainer)

    TOKENIZER_OUT.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(TOKENIZER_OUT))

    print(f"Tokenizer saved to: {TOKENIZER_OUT}")


if __name__ == "__main__":
    main()
