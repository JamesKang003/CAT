import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional

"""
PMC Data Ingestion Pipeline (V2)

This script constructs a training dataset from raw PubMed Central (PMC) XML articles.

Core responsibilities:
- Parse biomedical XML articles into structured text
- Remove structural artifacts (e.g., citation patterns)
- Construct coherent text chunks from article-local content
- Perform article-level train/validation/test splitting to prevent data leakage

Design rationale:
Chunks are generated within each article before splitting, ensuring that
no segments derived from the same source document are distributed across splits.
This enforces document-level independence during model evaluation.
"""

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "PMC001xxxxxx"
INTERIM_DIR = BASE_DIR / "data" / "interim"

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05
RANDOM_SEED = 42

MIN_PARAGRAPH_CHARS = 60

# Chunk-size heuristics chosen to preserve enough local context for training
# while keeping documents manageable for local hardware.
MIN_CHUNK_CHARS = 400
MAX_CHUNK_CHARS = 1400
TARGET_CHUNK_CHARS = 900

SEP_TOKEN = "[SEP]"


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces and trim the result."""
    return re.sub(r"\s+", " ", text).strip()


def get_local_tag(tag: str) -> str:
    """Remove the XML namespace prefix so tags can be compared consistently."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def iter_xml_files(root: Path) -> Iterable[Path]:
    """Yield all XML/NXML article files recursively under the raw PMC directory."""
    for ext in ("*.xml", "*.nxml"):
        yield from root.rglob(ext)


def clean_text(text: str) -> str:
    """
    Normalize extracted XML text for downstream chunking.

    This step removes common citation-style references and figure/table mentions,
    filters non-printable characters, and standardizes whitespace.
    """
    text = text.replace("\x00", " ")
    text = re.sub(r"\[\s*\d+(?:\s*[-,]\s*\d+)*\s*\]", " ", text)
    text = re.sub(r"\(\s*(?:fig|figure|table|tbl)\.?\s*[^)]*\)", " ", text, flags=re.I)
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\n")
    return normalize_whitespace(text)


def extract_title(root: ET.Element) -> Optional[str]:
    """Extract and clean the article title if one is present."""
    for elem in root.iter():
        if get_local_tag(elem.tag) == "article-title":
            title = clean_text("".join(elem.itertext()))
            if title:
                return title
    return None


def extract_paragraphs(section_elem: ET.Element) -> list[str]:
    """
    Extract paragraph text from an XML section.

    Only paragraphs above the minimum length threshold are kept so that very short
    fragments do not dominate the downstream chunking process.
    """
    paragraphs = []
    for elem in section_elem.iter():
        if get_local_tag(elem.tag) != "p":
            continue
        text = clean_text("".join(elem.itertext()))
        if len(text) >= MIN_PARAGRAPH_CHARS:
            paragraphs.append(text)
    return paragraphs


def extract_article_paragraphs(xml_path: Path) -> Optional[list[str]]:
    """
    Parse one PMC article and collect the title, abstract paragraphs, and body paragraphs.

    The returned list preserves article-local ordering and is later chunked without
    mixing content across source documents.
    """
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

    paragraphs = []

    if title:
        paragraphs.append(title)

    if abstract_elem is not None:
        abstract_paragraphs = extract_paragraphs(abstract_elem)
        if abstract_paragraphs:
            paragraphs.extend(abstract_paragraphs)

    if body_elem is not None:
        body_paragraphs = extract_paragraphs(body_elem)
        if body_paragraphs:
            paragraphs.extend(body_paragraphs)

    if not paragraphs:
        return None

    return paragraphs


def split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    """
    Split oversized paragraphs into sentence-based segments.

    This keeps extremely long paragraphs from creating oversized training chunks
    while preserving sentence boundaries when possible.
    """
    if len(paragraph) <= max_chars:
        return [paragraph]

    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    chunks = []
    current = []

    for sent in sentences:
        candidate = " ".join(current + [sent]).strip()
        if len(candidate) <= max_chars:
            current.append(sent)
        else:
            if current:
                chunks.append(" ".join(current).strip())
            current = [sent]

    if current:
        chunks.append(" ".join(current).strip())

    return [c for c in chunks if len(c) >= MIN_PARAGRAPH_CHARS]


def build_chunks(paragraphs: list[str]) -> list[str]:
    """
    Build chunked training documents from article-local paragraphs.

    The chunker first expands oversized paragraphs, then packs adjacent text into
    target-sized chunks separated by SEP tokens. This process is performed within
    a single article only, so chunk boundaries never cross source-document boundaries.
    """
    expanded = []
    for p in paragraphs:
        expanded.extend(split_long_paragraph(p, MAX_CHUNK_CHARS))

    chunks = []
    current_parts = []
    current_len = 0

    for p in expanded:
        addition = len(p) + (len(f" {SEP_TOKEN} ") if current_parts else 0)

        if current_parts and current_len + addition > TARGET_CHUNK_CHARS:
            chunk = f" {SEP_TOKEN} ".join(current_parts).strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
            current_parts = [p]
            current_len = len(p)
        else:
            current_parts.append(p)
            current_len += addition

        if current_len >= MAX_CHUNK_CHARS:
            chunk = f" {SEP_TOKEN} ".join(current_parts).strip()
            if len(chunk) >= MIN_CHUNK_CHARS:
                chunks.append(chunk)
            current_parts = []
            current_len = 0

    if current_parts:
        chunk = f" {SEP_TOKEN} ".join(current_parts).strip()
        if len(chunk) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)

    return chunks


def write_split(path: Path, docs: list[str]) -> None:
    """Write one split to disk as one normalized chunk per line."""
    with path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(normalize_whitespace(doc) + "\n")


def main() -> None:
    """
    Run the full PMC ingestion pipeline.

    Pipeline summary:
    1. Discover raw XML/NXML articles.
    2. Parse each article into article-local paragraphs.
    3. Build chunks without mixing content across source documents.
    4. Shuffle and split at the article level.
    5. Flatten chunks only after the train/val/test split is finalized.
    """
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(iter_xml_files(RAW_DIR))
    if not xml_files:
        raise FileNotFoundError(f"No .xml or .nxml files found under: {RAW_DIR}")

    print(f"Found {len(xml_files)} XML/NXML files under {RAW_DIR}")

    articles = []
    skipped = 0

    for xml_path in xml_files:
        paragraphs = extract_article_paragraphs(xml_path)
        if not paragraphs:
            skipped += 1
            continue

        # Chunk each article independently before splitting so that chunks derived
        # from the same source document remain grouped together.
        chunks = build_chunks(paragraphs)
        if not chunks:
            skipped += 1
            continue

        articles.append((xml_path.stem, chunks))

    random.seed(RANDOM_SEED)
    random.shuffle(articles)

    n_total = len(articles)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_articles = articles[:n_train]
    val_articles = articles[n_train : n_train + n_val]
    test_articles = articles[n_train + n_val :]

    # Flatten only after the article-level split has been assigned. This preserves
    # document independence across train/validation/test sets.
    train_docs = [chunk for _, chunks in train_articles for chunk in chunks]
    val_docs = [chunk for _, chunks in val_articles for chunk in chunks]
    test_docs = [chunk for _, chunks in test_articles for chunk in chunks]

    write_split(INTERIM_DIR / "train_raw.txt", train_docs)
    write_split(INTERIM_DIR / "val_raw.txt", val_docs)
    write_split(INTERIM_DIR / "test_raw.txt", test_docs)

    avg_chunks = sum(len(chunks) for _, chunks in articles) / len(articles)

    print("\n--- PMC Ingestion V2 Complete ---")
    print(f"Usable articles   : {n_total}")
    print(f"Skipped articles  : {skipped}")
    print(f"Avg chunks/article: {avg_chunks:.2f}")
    print(f"Train chunks      : {len(train_docs)}")
    print(f"Val chunks        : {len(val_docs)}")
    print(f"Test chunks       : {len(test_docs)}")


if __name__ == "__main__":
    main()
