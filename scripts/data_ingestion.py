import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------
# Project: Clinically-Aware Transformer (CAT)
# Module: PMC XML ingestion
# Purpose:
#   - Parse PMC full-text XML/NXML files
#   - Extract abstract + body paragraphs
#   - Clean noisy citation/reference text
#   - Split by ARTICLE (pseudo-subject) to reduce leakage
#   - Save one article per line into train/val/test raw text files
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "PMC001xxxxxx"
INTERIM_DIR = BASE_DIR / "data" / "interim"

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42

# Keep reasonably information-dense documents only.
MIN_PARAGRAPH_CHARS = 40
MIN_ARTICLE_CHARS = 800

# If one article is extremely long, cap it so one paper does not dominate.
MAX_ARTICLE_CHARS = 40000


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def iter_xml_files(root: Path) -> Iterable[Path]:
    for ext in ("*.xml", "*.nxml"):
        yield from root.rglob(ext)


def clean_text(text: str) -> str:
    """
    Clean PMC text while keeping medically meaningful prose.
    This is lighter than MIMIC de-id cleaning because PMC does not contain
    MIMIC-style [** ... **] placeholders.
    """
    text = text.replace("\x00", " ")

    # Remove bracketed numeric citations like [1], [2,3], [4-6]
    text = re.sub(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]", " ", text)

    # Remove parenthetical figure/table references
    text = re.sub(r"\(\s*(?:fig|figure|table|tbl)\.?\s*[^)]*\)", " ", text, flags=re.I)

    # Remove repeated punctuation noise
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove non-printable chars, keep normal ASCII-ish text
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\n")

    return normalize_whitespace(text)


def extract_title(root: ET.Element) -> Optional[str]:
    for elem in root.iter():
        if get_local_tag(elem.tag) == "article-title":
            title = clean_text("".join(elem.itertext()))
            if title:
                return title
    return None


def extract_section_paragraphs(section_elem: ET.Element) -> list[str]:
    paragraphs: list[str] = []

    for elem in section_elem.iter():
        if get_local_tag(elem.tag) != "p":
            continue

        text = clean_text("".join(elem.itertext()))
        if len(text) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(text)

    return paragraphs


def extract_article_text(xml_path: Path) -> Optional[str]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[WARN] Failed to parse {xml_path.name}: {e}")
        return None

    title = extract_title(root)

    abstract_elem = None
    body_elem = None

    for elem in root.iter():
        tag = get_local_tag(elem.tag)
        if tag == "abstract" and abstract_elem is None:
            abstract_elem = elem
        elif tag == "body" and body_elem is None:
            body_elem = elem

    parts: list[str] = []

    if title:
        parts.append(title)

    if abstract_elem is not None:
        abstract_paragraphs = extract_section_paragraphs(abstract_elem)
        if abstract_paragraphs:
            parts.append("Abstract: " + " ".join(abstract_paragraphs))

    if body_elem is not None:
        body_paragraphs = extract_section_paragraphs(body_elem)
        if body_paragraphs:
            parts.append("Body: " + " ".join(body_paragraphs))

    if not parts:
        return None

    article_text = "\n".join(parts)
    article_text = clean_text(article_text)

    if len(article_text) < MIN_ARTICLE_CHARS:
        return None

    if len(article_text) > MAX_ARTICLE_CHARS:
        article_text = article_text[:MAX_ARTICLE_CHARS].rsplit(" ", 1)[0]

    return article_text


def write_split(path: Path, docs: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(normalize_whitespace(doc) + "\n")


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(iter_xml_files(RAW_DIR))
    if not xml_files:
        raise FileNotFoundError(f"No .xml or .nxml files found under: {RAW_DIR}")

    print(f"Found {len(xml_files)} XML/NXML files under {RAW_DIR}")

    records: list[tuple[str, str]] = []
    skipped = 0

    for xml_path in xml_files:
        article_text = extract_article_text(xml_path)
        if article_text is None:
            skipped += 1
            continue
        records.append((xml_path.stem, article_text))

    if not records:
        raise RuntimeError("No usable PMC articles were extracted.")

    random.seed(RANDOM_SEED)
    random.shuffle(records)

    n_total = len(records)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)
    n_test = n_total - n_train - n_val

    train_docs = [text for _, text in records[:n_train]]
    val_docs = [text for _, text in records[n_train : n_train + n_val]]
    test_docs = [text for _, text in records[n_train + n_val :]]

    write_split(INTERIM_DIR / "train_raw.txt", train_docs)
    write_split(INTERIM_DIR / "val_raw.txt", val_docs)
    write_split(INTERIM_DIR / "test_raw.txt", test_docs)

    print("\n--- PMC Ingestion Complete ---")
    print(f"Usable articles : {n_total}")
    print(f"Skipped articles: {skipped}")
    print(f"Train           : {len(train_docs)}")
    print(f"Val             : {len(val_docs)}")
    print(f"Test            : {len(test_docs)}")

    if n_total > 0:
        avg_chars = sum(len(t) for _, t in records) / n_total
        print(f"Avg chars/article: {avg_chars:.1f}")


if __name__ == "__main__":
    main()
