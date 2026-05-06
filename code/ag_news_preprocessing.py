"""
AG News Multi-Class Text Classification — Preprocessing Pipeline
================================================================
HuggingFace dataset | 8000 samples | 4 classes | sklearn-compatible output
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import re
import string
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Download required NLTK resources (safe to call multiple times)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DATASET_NAME      = "ag_news"
TOTAL_SAMPLES     = 8_000
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
MIN_FREQ          = 2          # vocabulary pruning threshold
RARE_TOKEN        = "<UNK>"   # replacement for pruned tokens

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

STOP_WORDS = set(stopwords.words("english"))

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_ag_news(total_samples: int = TOTAL_SAMPLES):
    """
    Load AG News from HuggingFace and cap at *total_samples*.

    AG News ships with a fixed train/test split; we merge both splits first
    so our own 80-20 split is unbiased.

    Returns
    -------
    texts  : list[str]   — raw text strings (title + description)
    labels : list[int]   — integer class ids  {0, 1, 2, 3}
    """
    print("📥  Loading AG News from HuggingFace …")
    dataset = load_dataset(DATASET_NAME)

    # Merge train + test into one flat list
    all_texts  = (list(dataset["train"]["text"])  + list(dataset["test"]["text"]))
    all_labels = (list(dataset["train"]["label"]) + list(dataset["test"]["label"]))

    # Cap at requested size (stratified-safe: AG News is already balanced)
    texts  = all_texts[:total_samples]
    labels = all_labels[:total_samples]

    print(f"   ✓ Loaded {len(texts):,} samples")
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING STEPS  (each is a pure function for modularity)
# ─────────────────────────────────────────────────────────────────────────────

def lowercase(text: str) -> str:
    """
    Step 1 — Lowercasing
    Converts every character to lowercase so that 'Apple', 'apple', and
    'APPLE' are treated as the same token.  Without this step the model
    sees them as three distinct features.
    """
    return text.lower()


def remove_noise(text: str) -> str:
    """
    Step 2 — Noise removal
    Strips HTML tags, URLs, and non-alphanumeric punctuation that carry
    no semantic content, reducing vocabulary size.
    """
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)       # remaining punctuation
    text = re.sub(r"\s+", " ", text).strip()        # collapse whitespace
    return text


def tokenize(text: str) -> list[str]:
    """
    Step 3 — Tokenization
    Splits the cleaned string into a list of individual word tokens using
    NLTK's Punkt tokenizer, which handles contractions and edge-cases
    better than a naive str.split().

    Example:  "good morning" → ["good", "morning"]
    """
    return word_tokenize(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """
    Step 4 — Stopword removal
    Filters out high-frequency, low-information words such as 'the', 'is',
    'at', 'which'.  These words would otherwise dominate TF-IDF / BoW
    vectors and drown out discriminative content words.

    Uses NLTK's English stopword list (179 words).
    """
    return [tok for tok in tokens if tok not in STOP_WORDS]


def prune_vocabulary(
    corpus: list[list[str]],
    min_freq: int = MIN_FREQ,
    unk_token: str = RARE_TOKEN,
) -> tuple[list[list[str]], dict]:
    """
    Step 5 — Vocabulary pruning  (optional)
    Replaces tokens that appear fewer than *min_freq* times across the
    entire corpus with a special <UNK> placeholder.

    Why?  Hapax legomena (words seen only once) are usually typos or
    proper-noun noise.  Removing them:
      • shrinks feature space  → faster training
      • reduces overfitting   → better generalisation

    Returns the pruned corpus and the frequency vocabulary dict.
    """
    freq: Counter = Counter(tok for doc in corpus for tok in doc)
    vocab = {tok: cnt for tok, cnt in freq.items() if cnt >= min_freq}

    pruned = [
        [tok if tok in vocab else unk_token for tok in doc]
        for doc in corpus
    ]
    rare_count = sum(1 for tok, cnt in freq.items() if cnt < min_freq)
    print(f"   ✓ Vocabulary pruning: kept {len(vocab):,} tokens, "
          f"replaced {rare_count:,} rare tokens with '{unk_token}'")
    return pruned, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 3. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_corpus(
    texts: list[str],
    min_freq: int = MIN_FREQ,
    apply_vocab_pruning: bool = True,
) -> tuple[list[str], list[list[str]], dict]:
    """
    Applies the complete preprocessing pipeline to a list of raw texts.

    Parameters
    ----------
    texts              : raw text documents
    min_freq           : minimum token frequency for vocabulary retention
    apply_vocab_pruning: whether to apply Step 5

    Returns
    -------
    cleaned_texts : list[str]        — joined cleaned strings  (sklearn-ready)
    token_lists   : list[list[str]]  — token sequences per document
    vocab         : dict             — {token: frequency} after pruning
    """
    print("\n🔧  Preprocessing …")

    token_lists: list[list[str]] = []

    for text in texts:
        text = lowercase(text)          # Step 1
        text = remove_noise(text)       # Step 2
        tokens = tokenize(text)         # Step 3
        tokens = remove_stopwords(tokens)  # Step 4
        token_lists.append(tokens)

    print(f"   ✓ Steps 1-4 done for {len(token_lists):,} documents")

    vocab: dict = {}
    if apply_vocab_pruning:
        token_lists, vocab = prune_vocabulary(token_lists, min_freq)

    # Rejoin tokens into strings for sklearn vectorisers (TF-IDF, CountVectorizer …)
    cleaned_texts = [" ".join(tokens) for tokens in token_lists]

    return cleaned_texts, token_lists, vocab


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT  (sklearn-compatible)
# ─────────────────────────────────────────────────────────────────────────────

def split_data(
    texts: list[str],
    labels: list[int],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """
    Stratified 80-20 train/test split.
    Stratification ensures each class is proportionally represented in both
    splits, which is critical for balanced evaluation metrics.
    """
    return train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_class_distribution(labels: list[int], split_name: str = "Full"):
    """Pretty-prints class counts and percentages."""
    counter = Counter(labels)
    total   = len(labels)

    print(f"\n📊  Class distribution — {split_name} ({total:,} samples)")
    print(f"  {'Class':<12} {'Label':<12} {'Count':>7}  {'%':>6}")
    print("  " + "─" * 42)
    for cls_id in sorted(counter):
        cnt  = counter[cls_id]
        name = AG_NEWS_LABELS.get(cls_id, str(cls_id))
        print(f"  {cls_id:<12} {name:<12} {cnt:>7,}  {cnt/total*100:>5.1f}%")


def print_sample(
    cleaned_texts: list[str],
    token_lists:   list[list[str]],
    raw_texts:     list[str],
    labels:        list[int],
    n: int = 3,
):
    """Show a few examples before/after preprocessing."""
    print(f"\n🔍  Sample preview (first {n} documents)")
    for i in range(min(n, len(raw_texts))):
        print(f"\n  ── Sample {i+1}  [class: {AG_NEWS_LABELS[labels[i]]}] ──")
        print(f"  RAW     : {raw_texts[i][:120]} …")
        print(f"  CLEANED : {cleaned_texts[i][:120]} …")
        print(f"  TOKENS  : {token_lists[i][:15]} …")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load ──────────────────────────────────────────────────────────────────
    raw_texts, labels = load_ag_news(TOTAL_SAMPLES)

    # ── Class distribution (raw) ───────────────────────────────────────────
    print_class_distribution(labels, "Full dataset (raw)")

    # ── Preprocess ────────────────────────────────────────────────────────────
    cleaned_texts, token_lists, vocab = preprocess_corpus(
        raw_texts,
        min_freq=MIN_FREQ,
        apply_vocab_pruning=True,
    )

    # ── Sample preview ────────────────────────────────────────────────────────
    print_sample(cleaned_texts, token_lists, raw_texts, labels, n=3)

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = split_data(cleaned_texts, labels)

    print_class_distribution(y_train, "Train split")
    print_class_distribution(y_test,  "Test split")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n✅  Pipeline complete — sklearn-ready arrays")
    print(f"   X_train : {len(X_train):,} documents")
    print(f"   X_test  : {len(X_test):,}  documents")
    print(f"   Vocabulary size (after pruning, min_freq={MIN_FREQ}): "
          f"{len(vocab):,} tokens\n")

    return X_train, X_test, y_train, y_test, vocab


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, vocab = main()
