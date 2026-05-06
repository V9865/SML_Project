"""
AG News Multi-Class Text Classification — Transformer Embeddings
=================================================================
Extracts fixed-size sentence embeddings from pretrained transformers.

Models used  : distilbert-base-uncased  |  bert-base-uncased
Strategies   : [CLS] token embedding    |  Mean-pooled embedding
Mode         : Feature extraction ONLY  (model weights are FROZEN)
Device       : CUDA if available, else CPU
Output       : .npy files on disk (skip recomputation on re-runs)

Integrates with:
  ag_news_preprocessing.py  →  cleaned X_train / X_test text lists
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import time
import numpy as np
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    logging as hf_logging,
)

# Silence HuggingFace info messages (keep warnings/errors)
hf_logging.set_verbosity_error()

# ── Constants ─────────────────────────────────────────────────────────────────
MODELS = {
    "distilbert": "distilbert-base-uncased",
    "bert":       "bert-base-uncased",
}

MAX_LENGTH  = 128      # truncate / pad to this many tokens
BATCH_SIZE  = 32       # reduce to 16 if you hit GPU OOM
EMBED_DIR   = Path("embeddings")   # where .npy files are saved

EMBED_DIR.mkdir(exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚙️   Device : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"    GPU    : {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD TOKENIZER & MODEL
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_key: str):
    """
    Download (first run) or load from cache a pretrained tokenizer + model.

    The model is moved to the target device and set to eval() mode.
    No gradients are tracked anywhere in this pipeline — the weights
    are completely frozen.

    Parameters
    ----------
    model_key : one of {'distilbert', 'bert'}

    Returns
    -------
    tokenizer : HuggingFace fast tokenizer
    model     : pretrained transformer (frozen, on DEVICE)
    """
    model_name = MODELS[model_key]
    print(f"\n📥  Loading {model_name} …")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # AutoModel returns the base encoder WITHOUT a classification head.
    # This is exactly what we want for feature extraction.
    model = AutoModel.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()   # disables dropout — deterministic embeddings

    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ {model_name}  |  {n_params/1e6:.1f}M params  |  eval mode, frozen")

    return tokenizer, model


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRACT EMBEDDINGS  (batched, no-grad)
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(
    texts:     list[str],
    tokenizer,
    model,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
    split_name: str = "data",
) -> dict[str, np.ndarray]:
    """
    Run the transformer encoder over *texts* in mini-batches and collect
    two embedding strategies per document.

    Embedding strategies
    --------------------
    CLS token   : hidden state of the special [CLS] token prepended to every
                  input.  In BERT-style models the [CLS] representation is
                  trained to aggregate sequence-level information.
                  Shape per document: (hidden_size,)  e.g. (768,) for BERT

    Mean pool   : arithmetic mean of ALL non-padding token hidden states.
                  Averages information across every position rather than
                  relying on a single learned summary token.
                  Shape per document: (hidden_size,)

    Both strategies produce a single fixed-size vector per document,
    regardless of the original text length — perfect for feeding into
    sklearn classifiers.

    Parameters
    ----------
    texts      : list of preprocessed text strings
    tokenizer  : HuggingFace fast tokenizer (already loaded)
    model      : pretrained transformer in eval mode (already on DEVICE)
    batch_size : number of texts to process per forward pass
    max_length : tokenizer truncation/padding length
    split_name : label for progress messages (e.g. 'train', 'test')

    Returns
    -------
    dict with keys 'cls' and 'mean', each a float32 numpy array
    of shape (n_docs, hidden_size).
    """
    all_cls  = []
    all_mean = []
    n_batches = (len(texts) + batch_size - 1) // batch_size

    t0 = time.time()
    print(f"   Extracting embeddings for {len(texts):,} {split_name} docs "
          f"({n_batches} batches) …")

    for batch_idx in range(n_batches):
        batch_texts = texts[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        # Tokenize: returns input_ids, attention_mask (and token_type_ids for BERT)
        encoded = tokenizer(
            batch_texts,
            padding       = True,       # pad shorter seqs to longest in batch
            truncation    = True,       # clip sequences longer than max_length
            max_length    = max_length,
            return_tensors= "pt",       # return PyTorch tensors
        )
        # Move all tensors to the correct device (GPU or CPU)
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        # ── Forward pass ──────────────────────────────────────────────────────
        with torch.no_grad():
            # CRITICAL: torch.no_grad() disables gradient computation entirely.
            # This means:
            #   • No backward pass is possible → weights cannot be updated
            #   • ~2× faster inference
            #   • ~50% less memory (no activation graph stored)
            outputs = model(**encoded)

        # last_hidden_state : (batch, seq_len, hidden_size)
        hidden = outputs.last_hidden_state   # float32 on DEVICE

        # ── CLS embedding ─────────────────────────────────────────────────────
        # Position 0 is always [CLS] in BERT-style tokenizers.
        cls_emb = hidden[:, 0, :]            # (batch, hidden_size)

        # ── Mean-pool embedding ───────────────────────────────────────────────
        # attention_mask is 1 for real tokens, 0 for [PAD] tokens.
        # We expand it to (batch, seq_len, 1) to broadcast across hidden_size,
        # then zero-out padding positions before averaging.
        mask        = encoded["attention_mask"].unsqueeze(-1).float()  # (B, S, 1)
        sum_hidden  = (hidden * mask).sum(dim=1)                       # (B, H)
        token_count = mask.sum(dim=1).clamp(min=1e-9)                  # (B, 1)
        mean_emb    = sum_hidden / token_count                          # (B, H)

        # Move back to CPU for numpy conversion (GPU tensors can't → numpy directly)
        all_cls.append(cls_emb.cpu().numpy())
        all_mean.append(mean_emb.cpu().numpy())

        # Progress indicator every 10 batches
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
            elapsed = time.time() - t0
            pct     = (batch_idx + 1) / n_batches * 100
            print(f"     [{batch_idx+1:>3}/{n_batches}]  {pct:5.1f}%  "
                  f"({elapsed:.1f}s elapsed)")

    cls_array  = np.vstack(all_cls).astype(np.float32)   # (n_docs, hidden_size)
    mean_array = np.vstack(all_mean).astype(np.float32)

    print(f"   ✓ Done in {time.time()-t0:.1f}s  |  "
          f"CLS shape: {cls_array.shape}  |  Mean shape: {mean_array.shape}")

    return {"cls": cls_array, "mean": mean_array}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SAVE / LOAD  (skip recomputation)
# ─────────────────────────────────────────────────────────────────────────────

def _npy_path(model_key: str, strategy: str, split: str) -> Path:
    """Return deterministic file path for a given (model, strategy, split)."""
    return EMBED_DIR / f"{model_key}_{strategy}_{split}.npy"


def save_embeddings(embeddings: dict, model_key: str, split: str):
    """Persist CLS and mean embeddings to disk as .npy files."""
    for strategy, array in embeddings.items():
        path = _npy_path(model_key, strategy, split)
        np.save(path, array)
        print(f"   💾  Saved: {path}  ({array.nbytes / 1024**2:.1f} MB)")


def load_embeddings_from_disk(model_key: str, split: str) -> dict | None:
    """
    Return embeddings from disk if all expected files exist, else None.
    This lets you skip the expensive forward pass on re-runs.
    """
    result = {}
    for strategy in ("cls", "mean"):
        path = _npy_path(model_key, strategy, split)
        if not path.exists():
            return None
        result[strategy] = np.load(path)
    print(f"   ⚡  Loaded from cache: {model_key} / {split}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. PUBLIC API — get_embeddings
# ─────────────────────────────────────────────────────────────────────────────

def get_embeddings(
    X_train: list[str],
    X_test:  list[str],
    model_key:  str  = "distilbert",
    batch_size: int  = BATCH_SIZE,
    max_length: int  = MAX_LENGTH,
    force_recompute: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Main entry point.  Returns embeddings for both splits, using disk cache
    when available.

    Parameters
    ----------
    X_train, X_test   : preprocessed text lists from ag_news_preprocessing
    model_key         : 'distilbert' or 'bert'
    force_recompute   : ignore disk cache and recompute from scratch

    Returns
    -------
    {
        'train': {'cls': np.ndarray (n_train, H), 'mean': np.ndarray (n_train, H)},
        'test' : {'cls': np.ndarray (n_test,  H), 'mean': np.ndarray (n_test,  H)},
    }

    Usage (integrated pipeline)
    ---------------------------
    from ag_news_preprocessing import main as preprocess
    from ag_news_embeddings    import get_embeddings

    X_train, X_test, y_train, y_test, vocab = preprocess()

    emb = get_embeddings(X_train, X_test, model_key='distilbert')
    # emb['train']['cls']  →  (6400, 768)
    # emb['train']['mean'] →  (6400, 768)
    # emb['test']['cls']   →  (1600, 768)
    """
    print(f"\n{'='*60}")
    print(f"  TRANSFORMER EMBEDDINGS  —  {MODELS[model_key]}")
    print(f"{'='*60}")

    result = {}
    tokenizer, model = None, None   # lazy-load only if cache miss

    for split, texts in [("train", X_train), ("test", X_test)]:
        if not force_recompute:
            cached = load_embeddings_from_disk(model_key, split)
            if cached is not None:
                result[split] = cached
                continue

        # Cache miss — load model on first need
        if tokenizer is None:
            tokenizer, model = load_model(model_key)

        embs = extract_embeddings(
            texts, tokenizer, model,
            batch_size = batch_size,
            max_length = max_length,
            split_name = split,
        )
        save_embeddings(embs, model_key, split)
        result[split] = embs

    _print_embedding_summary(result, model_key)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 5. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def _print_embedding_summary(result: dict, model_key: str):
    """Print shapes, memory, and L2-norm stats for all embedding arrays."""
    print(f"\n{'─'*60}")
    print(f"  EMBEDDING SUMMARY — {model_key.upper()}")
    print(f"{'─'*60}")
    print(f"  {'Split':<8} {'Strategy':<8} {'Shape':<20} {'MB':>6}  {'‖x‖₂ mean':>10}")
    print(f"  {'─'*55}")
    for split in ("train", "test"):
        for strat in ("cls", "mean"):
            arr  = result[split][strat]
            mb   = arr.nbytes / 1024**2
            norm = np.linalg.norm(arr, axis=1).mean()
            print(f"  {split:<8} {strat:<8} {str(arr.shape):<20} {mb:>6.1f}  {norm:>10.4f}")
    print()


def compare_strategies(result: dict, y_train, y_test):
    """
    Quick sanity check: train a LinearSVC on each embedding type and
    print accuracy — no hyperparameter tuning, just a baseline comparison.
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import normalize

    print(f"\n{'─'*60}")
    print("  QUICK ACCURACY CHECK  (LinearSVC, no tuning)")
    print(f"{'─'*60}")
    print(f"  {'Strategy':<30}  {'Accuracy':>9}")
    print(f"  {'─'*42}")

    for strat in ("cls", "mean"):
        X_tr = normalize(result["train"][strat])
        X_te = normalize(result["test"][strat])

        clf = LinearSVC(max_iter=2000, C=1.0)
        clf.fit(X_tr, y_train)
        acc = clf.score(X_te, y_test)

        print(f"  {strat:<30}  {acc*100:>8.2f}%")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Synthetic corpus (mirrors preprocessing output) ───────────────────────
    # On your machine replace this block with:
    #   from ag_news_preprocessing import main as preprocess
    #   X_train, X_test, y_train, y_test, vocab = preprocess()
    from sklearn.model_selection import train_test_split

    docs = [
        "us president signs new trade deal european union economic growth",
        "stock market rally federal reserve interest rate decision investors",
        "soccer world cup brazil argentina quarter final goal penalty",
        "nasa moon mission astronaut launch rocket orbit earth",
        "apple iphone new model release processor chip battery technology",
        "olympic games paris france swimming athletics world record medal",
        "china trade war tariff technology export semiconductor chip supply",
        "football nfl super bowl touchdown quarterback touchdown pass yards",
        "climate change global warming carbon emission renewable energy solar",
        "election vote president candidate campaign poll democrat republican",
        "basketball nba playoffs lakers celtics championship finals score win",
        "inflation consumer price index federal reserve monetary policy rate",
        "spacex starship launch elon musk orbit reusable rocket booster test",
        "tennis wimbledon final grand slam serve ace match point set break",
        "semiconductor shortage chip supply chain manufacturing taiwan plant",
        "artificial intelligence machine learning model training data gpu nvidia",
        "cricket ipl match india pakistan wicket boundary run score over",
        "oil price opec barrel crude petroleum market demand supply output",
        "cycling tour france stage winner yellow jersey time trial climb hill",
        "quantum computing qubit processor error correction algorithm ibm",
    ] * 400  # 8 000 docs

    labels = ([0] * 100 + [1] * 100 + [2] * 100 + [3] * 100) * 20

    X_train, X_test, y_train, y_test = train_test_split(
        docs, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # ── Extract embeddings for both models ────────────────────────────────────
    all_embeddings = {}

    for model_key in ["distilbert", "bert"]:
        emb = get_embeddings(
            X_train, X_test,
            model_key       = model_key,
            batch_size      = BATCH_SIZE,
            max_length      = MAX_LENGTH,
            force_recompute = False,   # set True to ignore cache
        )
        all_embeddings[model_key] = emb

        # Quick accuracy comparison between CLS and mean-pool
        compare_strategies(emb, y_train, y_test)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n✅  All embeddings extracted and saved to ./embeddings/")
    print("    Files generated:")
    for path in sorted(EMBED_DIR.glob("*.npy")):
        mb = path.stat().st_size / 1024**2
        print(f"    • {path.name:<45}  {mb:>6.1f} MB")

    print("\n    Ready for classifier training (Step 4 of pipeline):")
    print("    emb['train']['cls']   →  np.ndarray (n_train, 768)")
    print("    emb['train']['mean']  →  np.ndarray (n_train, 768)")
    print("    emb['test']['cls']    →  np.ndarray (n_test,  768)")
