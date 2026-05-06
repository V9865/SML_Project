"""
AG News Multi-Class Text Classification — Feature Extraction
=============================================================
Converts preprocessed text into machine-learning-ready feature matrices.

Representations
---------------
  BoW   : Bag-of-Words  (unigram, CountVectorizer)
  TF-IDF: Term Frequency-Inverse Document Frequency  (bigram, TfidfVectorizer)

Integrates directly with the output of ag_news_preprocessing.py
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────
BOW_MAX_FEATURES   = 20_000   # keep top-N tokens by frequency
TFIDF_MAX_FEATURES = 20_000   # keep top-N tokens by TF-IDF score
MIN_DF             = 2        # ignore tokens in fewer than 2 documents
N_SAMPLE_FEATURES  = 20       # how many feature names to print as a sample


# ─────────────────────────────────────────────────────────────────────────────
# 1. BAG-OF-WORDS  (unigram)
# ─────────────────────────────────────────────────────────────────────────────

def build_bow(
    X_train: list[str],
    X_test:  list[str],
    max_features: int = BOW_MAX_FEATURES,
    min_df: int       = MIN_DF,
):
    """
    Bag-of-Words — CountVectorizer (unigram)
    ----------------------------------------
    Each document becomes a sparse vector of raw word counts.

    Representation
    ~~~~~~~~~~~~~~
    Vocabulary V is built from X_train only (no data leakage).
    For a document d and token t:

        bow(d, t) = count of t in d

    Parameters
    ----------
    max_features : Keep only the top-N most frequent tokens across the corpus.
                   Limits memory and speeds up training.
    min_df       : Ignore tokens that appear in fewer than min_df documents.
                   Works hand-in-hand with vocabulary pruning from Step 5.

    Returns
    -------
    X_bow_train : sparse matrix  (n_train_docs × vocab_size)
    X_bow_test  : sparse matrix  (n_test_docs  × vocab_size)
    bow_vec     : fitted CountVectorizer  (use to inspect vocabulary)
    """
    bow_vec = CountVectorizer(
        ngram_range  = (1, 1),          # unigrams only
        max_features = max_features,
        min_df       = min_df,
        token_pattern= r"(?u)\b\w+\b",  # keeps single-char tokens
    )

    X_bow_train = bow_vec.fit_transform(X_train)   # fit on train only
    X_bow_test  = bow_vec.transform(X_test)         # apply to test (no leakage)

    return X_bow_train, X_bow_test, bow_vec


# ─────────────────────────────────────────────────────────────────────────────
# 2. TF-IDF  (bigram)
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf(
    X_train: list[str],
    X_test:  list[str],
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int       = MIN_DF,
):
    """
    TF-IDF — TfidfVectorizer (unigram + bigram)
    --------------------------------------------
    Each document becomes a sparse vector of TF-IDF weighted scores.

    Formula (sklearn default — smooth IDF)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        TF(t, d)   = count(t in d) / len(d)          # relative frequency
        IDF(t)     = log((1 + N) / (1 + df(t))) + 1  # inverse document freq
        TF-IDF     = TF × IDF

    Bigrams (ngram_range=(1,2))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Include both single tokens AND consecutive pairs, e.g.:
        "new york"  →  features: "new", "york", "new york"
    This lets the model capture multi-word phrases and local context that
    unigrams cannot represent.

    Parameters / Returns: same semantics as build_bow()
    """
    tfidf_vec = TfidfVectorizer(
        ngram_range  = (1, 2),          # unigrams + bigrams
        max_features = max_features,
        min_df       = min_df,
        sublinear_tf = True,            # replace TF with 1+log(TF) to dampen
                                        # the effect of very frequent tokens
        token_pattern= r"(?u)\b\w+\b",
    )

    X_tfidf_train = tfidf_vec.fit_transform(X_train)
    X_tfidf_test  = tfidf_vec.transform(X_test)

    return X_tfidf_train, X_tfidf_test, tfidf_vec


# ─────────────────────────────────────────────────────────────────────────────
# 3. DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_matrix_stats(matrix, name: str):
    """Print shape, density, and memory of a sparse feature matrix."""
    n_docs, n_feats = matrix.shape
    nnz      = matrix.nnz
    density  = nnz / (n_docs * n_feats) * 100
    mem_mb   = matrix.data.nbytes / 1024 / 1024

    print(f"\n  {'─'*50}")
    print(f"  {name}")
    print(f"  {'─'*50}")
    print(f"  Shape        : {n_docs:,} docs  ×  {n_feats:,} features")
    print(f"  Non-zeros    : {nnz:,}  ({density:.3f}% dense)")
    print(f"  Memory (data): {mem_mb:.2f} MB")


def print_sample_features(vectorizer, label: str, n: int = N_SAMPLE_FEATURES):
    """Show a sample of the learned feature names."""
    names = vectorizer.get_feature_names_out()
    step  = max(1, len(names) // n)
    sample = names[::step][:n]

    print(f"\n  Sample feature names — {label}")
    print(f"  {list(sample)}")


def print_feature_summary(
    X_bow_train, X_bow_test, bow_vec,
    X_tfidf_train, X_tfidf_test, tfidf_vec,
):
    """Consolidated diagnostic output for both vectorisers."""
    print("\n" + "="*60)
    print("  FEATURE EXTRACTION SUMMARY")
    print("="*60)

    print("\n📦  MATRIX SHAPES")
    print_matrix_stats(X_bow_train,   "BoW  — Train")
    print_matrix_stats(X_bow_test,    "BoW  — Test")
    print_matrix_stats(X_tfidf_train, "TF-IDF — Train")
    print_matrix_stats(X_tfidf_test,  "TF-IDF — Test")

    print("\n\n🔤  SAMPLE FEATURE NAMES")
    print_sample_features(bow_vec,   "BoW  (unigrams)")
    print_sample_features(tfidf_vec, "TF-IDF (unigrams + bigrams)")

    print("\n\n📊  TOP 15 BoW TOKENS (by corpus frequency)")
    freq = np.asarray(X_bow_train.sum(axis=0)).flatten()
    top_idx = freq.argsort()[::-1][:15]
    names   = bow_vec.get_feature_names_out()
    print(f"  {'Token':<20}  {'Count':>8}")
    print(f"  {'─'*30}")
    for i in top_idx:
        print(f"  {names[i]:<20}  {int(freq[i]):>8,}")

    print("\n\n📊  TOP 15 TF-IDF FEATURES (by mean weight)")
    weight = np.asarray(X_tfidf_train.mean(axis=0)).flatten()
    top_idx = weight.argsort()[::-1][:15]
    names   = tfidf_vec.get_feature_names_out()
    print(f"  {'Feature':<30}  {'Mean TF-IDF':>12}")
    print(f"  {'─'*44}")
    for i in top_idx:
        print(f"  {names[i]:<30}  {weight[i]:>12.6f}")

    print("\n" + "="*60)
    print("  ✅  Feature matrices ready for classifier training")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN  (standalone demo with synthetic data; use real pipeline in practice)
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    X_train: list[str],
    X_test:  list[str],
    bow_max_features:   int = BOW_MAX_FEATURES,
    tfidf_max_features: int = TFIDF_MAX_FEATURES,
    min_df: int             = MIN_DF,
    verbose: bool           = True,
):
    """
    Public API — takes preprocessed text lists, returns both feature matrices.

    Usage
    -----
    Plug directly into the preprocessing pipeline:

        from ag_news_preprocessing import main as preprocess
        from ag_news_features import extract_features

        X_train, X_test, y_train, y_test, vocab = preprocess()
        X_bow_tr, X_bow_te, X_tfidf_tr, X_tfidf_te, bow_vec, tfidf_vec = \\
            extract_features(X_train, X_test)

    Returns
    -------
    X_bow_train, X_bow_test       : sparse BoW matrices
    X_tfidf_train, X_tfidf_test   : sparse TF-IDF matrices
    bow_vec, tfidf_vec            : fitted vectorisers (for inspection / reuse)
    """
    print("🔢  Building feature matrices …")

    print("   • Bag-of-Words  (unigram, CountVectorizer) …")
    X_bow_train, X_bow_test, bow_vec = build_bow(
        X_train, X_test,
        max_features = bow_max_features,
        min_df       = min_df,
    )

    print("   • TF-IDF  (bigram, TfidfVectorizer) …")
    X_tfidf_train, X_tfidf_test, tfidf_vec = build_tfidf(
        X_train, X_test,
        max_features = tfidf_max_features,
        min_df       = min_df,
    )

    if verbose:
        print_feature_summary(
            X_bow_train, X_bow_test, bow_vec,
            X_tfidf_train, X_tfidf_test, tfidf_vec,
        )

    return (
        X_bow_train, X_bow_test,
        X_tfidf_train, X_tfidf_test,
        bow_vec, tfidf_vec,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo (synthetic corpus matching preprocessing output format)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic corpus — replace with actual pipeline output on your machine:
    #   from ag_news_preprocessing import main as preprocess
    #   X_train, X_test, y_train, y_test, vocab = preprocess()
    #   extract_features(X_train, X_test)

    synthetic_docs = [
        "us president signs new trade deal european union economic growth",
        "stock market rally federal reserve interest rate decision investors",
        "soccer world cup brazil argentina quarter final goal penalty",
        "nasa moon mission astronaut launch rocket orbit earth",
        "apple iphone new model release processor chip battery technology",
        "olympic games paris france swimming athletics world record medal",
        "china us trade war tariff technology export semiconductor chip",
        "football nfl super bowl touchdown quarterback touchdown pass",
        "climate change global warming carbon emission renewable energy solar",
        "election vote president candidate campaign poll democrat republican",
        "basketball nba playoffs lakers celtics championship finals score",
        "inflation consumer price index federal reserve monetary policy rate",
        "spacex starship launch elon musk orbit reusable rocket booster",
        "tennis wimbledon final grand slam serve ace match point set",
        "semiconductor shortage chip supply chain manufacturing taiwan",
        "artificial intelligence machine learning model training data gpu",
        "cricket ipl match india pakistan wicket boundary run score",
        "oil price opec barrel crude petroleum market demand supply",
        "cycling tour france stage winner yellow jersey time trial climb",
        "quantum computing qubit processor error correction algorithm",
    ] * 400   # scale to 8 000 docs

    labels_demo = ([0]*100 + [1]*100 + [2]*100 + [3]*100) * 20

    X_tr, X_te, y_tr, y_te = train_test_split(
        synthetic_docs, labels_demo,
        test_size=0.2, random_state=42, stratify=labels_demo,
    )

    (X_bow_train, X_bow_test,
     X_tfidf_train, X_tfidf_test,
     bow_vec, tfidf_vec) = extract_features(X_tr, X_te)
