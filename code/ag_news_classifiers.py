"""
AG News Multi-Class Text Classification — Model Training
=========================================================
Trains four classical ML models on two feature spaces:
  • Sparse features  : TF-IDF (from ag_news_features.py)
  • Dense embeddings : BERT / DistilBERT (from ag_news_embeddings.py)

Models
------
  1. Logistic Regression  — linear probabilistic classifier
  2. SVM Linear           — maximum-margin hyperplane (linear kernel)
  3. SVM RBF              — maximum-margin hyperplane (RBF / Gaussian kernel)
  4. KNN                  — k-nearest neighbours (distance-based)
  5. KMeans               — unsupervised clustering → majority-vote label mapping

All results are stored in a nested dict for easy downstream consumption.
"""

# ── Imports ────────────────────────────────────────────────────────────────
import numpy as np
from collections import Counter

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC, SVC
from sklearn.neighbors       import KNeighborsClassifier
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import LabelEncoder, Normalizer
from sklearn.pipeline        import Pipeline


# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
N_CLASSES      = 4
KNN_NEIGHBORS  = 7       # odd number avoids ties
KMEANS_INIT    = "k-means++"
KMEANS_N_INIT  = 10


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(X):
    """Convert sparse matrix → dense ndarray if needed."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.array(X)


def _kmeans_cluster_to_label(cluster_ids: np.ndarray,
                              true_labels:  np.ndarray,
                              n_clusters:   int) -> np.ndarray:
    """
    Map KMeans cluster indices to class labels via majority voting.

    For each cluster k, find the most common true label among the
    training examples assigned to that cluster.  This gives a
    deterministic cluster→label mapping that is then applied to
    both train and test predictions.

    Why majority voting?
    KMeans output is arbitrary permutation-invariant indices
    (cluster 0 is not class 0).  Majority voting aligns them
    with the ground-truth classes in a single pass.
    """
    mapping = {}
    for k in range(n_clusters):
        mask = cluster_ids == k
        if mask.sum() == 0:
            mapping[k] = 0          # empty cluster → fallback label
        else:
            mapping[k] = Counter(true_labels[mask]).most_common(1)[0][0]
    return np.array([mapping[c] for c in cluster_ids])


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOGISTIC REGRESSION
# ─────────────────────────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, X_test):
    """
    Logistic Regression  (multinomial, L2 regularisation, lbfgs solver)
    ─────────────────────────────────────────────────────────────────────
    Decision boundary : linear
    Outputs           : class probabilities via softmax
    Strengths         : fast, interpretable, well-calibrated probabilities,
                        excellent on high-dimensional sparse TF-IDF features
    Hyperparameters   : C=1.0 (inverse regularisation strength)
    """
    print("   • Logistic Regression …", end=" ", flush=True)
    clf = LogisticRegression(
        solver       = "lbfgs",
        max_iter     = 2_000,
        C            = 1.0,
        random_state = RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)
    print("done")
    return clf, preds, proba


# ─────────────────────────────────────────────────────────────────────────────
# 2. SVM — LINEAR
# ─────────────────────────────────────────────────────────────────────────────

def train_svm_linear(X_train, y_train, X_test):
    """
    Linear SVM  (LinearSVC — optimised for large sparse inputs)
    ────────────────────────────────────────────────────────────
    Decision boundary : linear hyperplane
    Why SVM works well for text
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Text feature spaces are extremely high-dimensional (20k+ features) but
    most documents are linearly separable in that space.  SVM finds the
    maximum-margin hyperplane, which:
      • generalises well with few training examples per dimension
      • is robust to irrelevant features (sparse activations)
      • handles imbalanced class overlap better than LR
    LinearSVC uses liblinear — far faster than kernel SVC for large corpora.
    """
    print("   • SVM (Linear) …", end=" ", flush=True)
    clf = LinearSVC(
        C            = 1.0,
        max_iter     = 2_000,
        random_state = RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("done")
    return clf, preds


# ─────────────────────────────────────────────────────────────────────────────
# 3. SVM — RBF KERNEL
# ─────────────────────────────────────────────────────────────────────────────

def train_svm_rbf(X_train, y_train, X_test):
    """
    RBF SVM  (Gaussian kernel — maps to infinite-dimensional feature space)
    ────────────────────────────────────────────────────────────────────────
    Decision boundary : non-linear (radial basis functions)
    Use case          : dense embeddings (BERT CLS / mean-pool)
    Note              : Requires dense input + L2 normalisation.
                        Expensive O(n²) — subsample large datasets.

    When does RBF beat Linear?
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    For TF-IDF (sparse + high-dim) : linear kernel dominates
    For BERT embeddings (dense 768) : RBF can capture non-linear structure
    """
    print("   • SVM (RBF) …", end=" ", flush=True)
    X_tr = _to_dense(X_train)
    X_te = _to_dense(X_test)

    # L2 normalise — makes RBF kernel equivalent to cosine similarity
    norm  = Normalizer(norm="l2")
    X_tr  = norm.fit_transform(X_tr)
    X_te  = norm.transform(X_te)

    clf = SVC(
        kernel       = "rbf",
        C            = 10.0,
        gamma        = "scale",
        decision_function_shape = "ovr",
        random_state = RANDOM_STATE,
    )
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    print("done")
    return clf, preds


# ─────────────────────────────────────────────────────────────────────────────
# 4. KNN
# ─────────────────────────────────────────────────────────────────────────────

def train_knn(X_train, y_train, X_test, n_neighbors: int = KNN_NEIGHBORS):
    """
    K-Nearest Neighbours  (cosine distance, brute-force search)
    ────────────────────────────────────────────────────────────
    Decision boundary : non-parametric (Voronoi regions)
    Distance metric   : cosine — better than Euclidean for TF-IDF/embeddings
                        because it ignores document length differences
    Strengths         : zero training cost, naturally multi-class,
                        handles complex boundaries
    Weaknesses        : slow inference O(n·d), sensitive to irrelevant dims
    """
    print(f"   • KNN (k={n_neighbors}) …", end=" ", flush=True)
    X_tr = _to_dense(X_train)
    X_te = _to_dense(X_test)

    clf = KNeighborsClassifier(
        n_neighbors = n_neighbors,
        metric      = "cosine",
        algorithm   = "brute",
        n_jobs      = -1,
    )
    clf.fit(X_tr, y_train)
    preds = clf.predict(X_te)
    print("done")
    return clf, preds


# ─────────────────────────────────────────────────────────────────────────────
# 5. KMEANS  (unsupervised → majority-vote label mapping)
# ─────────────────────────────────────────────────────────────────────────────

def train_kmeans(X_train, y_train, X_test,
                 n_clusters: int = N_CLASSES):
    """
    KMeans Clustering  (unsupervised → supervised via majority voting)
    ─────────────────────────────────────────────────────────────────
    KMeans partitions data into k=4 clusters by minimising within-cluster
    inertia (sum of squared distances to cluster centroid).

    Label mapping via majority voting
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    After fitting, each cluster k is assigned the true class that is
    most common among the training examples in that cluster.
    This mapping is then applied to test predictions.

    Strengths  : no labelled data needed during training
    Weaknesses : assumes spherical clusters, sensitive to initialisation,
                 cluster ≠ class boundaries (lower accuracy expected)
    """
    print(f"   • KMeans (k={n_clusters}) …", end=" ", flush=True)
    X_tr = _to_dense(X_train)
    X_te = _to_dense(X_test)

    km = KMeans(
        n_clusters   = n_clusters,
        init         = KMEANS_INIT,
        n_init       = KMEANS_N_INIT,
        random_state = RANDOM_STATE,
        max_iter     = 300,
    )
    km.fit(X_tr)

    train_cluster_ids = km.labels_
    test_cluster_ids  = km.predict(X_te)
    y_train_arr       = np.array(y_train)

    # Build cluster→label mapping from training assignments
    mapping = {}
    for k in range(n_clusters):
        mask = train_cluster_ids == k
        if mask.sum() == 0:
            mapping[k] = 0
        else:
            mapping[k] = Counter(y_train_arr[mask]).most_common(1)[0][0]

    test_preds = np.array([mapping[c] for c in test_cluster_ids])

    print("done")
    return km, test_preds, mapping


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — train_all_models
# ─────────────────────────────────────────────────────────────────────────────

def train_all_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_label: str = "tfidf",
    skip_rbf: bool     = False,
) -> dict:
    """
    Train all five models on the given feature matrix.

    Parameters
    ----------
    X_train, X_test   : sparse or dense feature matrices
    y_train, y_test   : integer label lists / arrays
    feature_label     : identifier stored in results dict ("tfidf" or "bert")
    skip_rbf          : set True to skip RBF SVM on large datasets

    Returns
    -------
    results : dict  {model_name: {predictions, model, ...}}

    Structure
    ~~~~~~~~~
    results = {
        "logistic_regression": {"model": clf, "preds": array, "proba": array},
        "svm_linear":          {"model": clf, "preds": array},
        "svm_rbf":             {"model": clf, "preds": array},
        "knn":                 {"model": clf, "preds": array},
        "kmeans":              {"model": km,  "preds": array, "mapping": dict},
    }
    """
    print(f"\n🤖  Training models on [{feature_label}] features …")

    results = {}
    y_train_arr = np.array(y_train)
    y_test_arr  = np.array(y_test)

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    clf, preds, proba = train_logistic_regression(X_train, y_train_arr, X_test)
    results["logistic_regression"] = {
        "model": clf, "preds": preds, "proba": proba,
        "feature": feature_label,
    }

    # ── 2. SVM Linear ─────────────────────────────────────────────────────────
    clf, preds = train_svm_linear(X_train, y_train_arr, X_test)
    results["svm_linear"] = {
        "model": clf, "preds": preds,
        "feature": feature_label,
    }

    # ── 3. SVM RBF ────────────────────────────────────────────────────────────
    if not skip_rbf:
        clf, preds = train_svm_rbf(X_train, y_train_arr, X_test)
        results["svm_rbf"] = {
            "model": clf, "preds": preds,
            "feature": feature_label,
        }
    else:
        print("   • SVM (RBF) … skipped")

    # ── 4. KNN ────────────────────────────────────────────────────────────────
    clf, preds = train_knn(X_train, y_train_arr, X_test)
    results["knn"] = {
        "model": clf, "preds": preds,
        "feature": feature_label,
    }

    # ── 5. KMeans ─────────────────────────────────────────────────────────────
    km, preds, mapping = train_kmeans(X_train, y_train_arr, X_test)
    results["kmeans"] = {
        "model": km, "preds": preds, "mapping": mapping,
        "feature": feature_label,
    }

    print(f"   ✅  All models trained on [{feature_label}]")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED RUNNER — both feature spaces
# ─────────────────────────────────────────────────────────────────────────────

def run_all(
    X_tfidf_train, X_tfidf_test,
    X_bert_train,  X_bert_test,
    y_train, y_test,
    skip_rbf_tfidf: bool = True,
) -> dict:
    """
    Train all models on both TF-IDF and BERT feature spaces.

    Returns
    -------
    all_results : {
        "tfidf": { model_name: result_dict, … },
        "bert":  { model_name: result_dict, … },
    }

    Usage
    -----
    from ag_news_preprocessing import main as preprocess
    from ag_news_features      import extract_features
    from ag_news_embeddings    import get_embeddings
    from ag_news_classifiers   import run_all

    X_train, X_test, y_train, y_test, vocab = preprocess()
    _, _, X_tfidf_tr, X_tfidf_te, _, _ = extract_features(X_train, X_test)
    emb = get_embeddings(X_train, X_test, model_key='distilbert')
    X_bert_tr = emb['train']['cls']
    X_bert_te = emb['test']['cls']

    all_results = run_all(X_tfidf_tr, X_tfidf_te, X_bert_tr, X_bert_te, y_train, y_test)
    """
    all_results = {}

    all_results["tfidf"] = train_all_models(
        X_tfidf_train, X_tfidf_test, y_train, y_test,
        feature_label  = "tfidf",
        skip_rbf       = skip_rbf_tfidf,   # linear dominates for sparse
    )

    all_results["bert"] = train_all_models(
        X_bert_train, X_bert_test, y_train, y_test,
        feature_label  = "bert",
        skip_rbf       = False,            # RBF is useful for dense embeddings
    )

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def print_predictions_summary(all_results: dict, y_test):
    """Quick sanity-check table: model × feature → prediction distribution."""
    y_test_arr = np.array(y_test)
    print("\n" + "="*65)
    print("  PREDICTIONS SUMMARY")
    print("="*65)
    print(f"  {'Feature':<8}  {'Model':<22}  {'Pred dist (0/1/2/3)'}")
    print("  " + "─"*60)

    for feat, models in all_results.items():
        for model_name, info in models.items():
            preds = info["preds"]
            dist  = Counter(preds)
            dist_str = "  ".join(f"{dist.get(i,0):>4}" for i in range(4))
            print(f"  {feat:<8}  {model_name:<22}  {dist_str}")
        print()

    print("="*65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Standalone demo (synthetic data — replace with real pipeline on your machine)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from sklearn.datasets        import make_classification
    from sklearn.model_selection import train_test_split
    from scipy.sparse            import csr_matrix

    print("🧪  Standalone demo with synthetic data")
    print("    (On your machine: use real TF-IDF + BERT embedding arrays)\n")

    # Synthetic sparse TF-IDF-like features (20 000 dim, sparse)
    X_raw, y = make_classification(
        n_samples=2_000, n_features=500, n_informative=50,
        n_classes=4, n_clusters_per_class=1, random_state=42,
    )
    X_tfidf = csr_matrix(np.abs(X_raw))  # simulate sparse non-negative TF-IDF
    X_bert  = X_raw                      # dense 500-dim (simulates 768-dim BERT)

    X_tfidf_tr, X_tfidf_te, X_bert_tr, X_bert_te, y_tr, y_te = train_test_split(
        X_tfidf, X_bert, y, test_size=0.2, random_state=42, stratify=y,
    )

    all_results = run_all(
        X_tfidf_tr, X_tfidf_te,
        X_bert_tr,  X_bert_te,
        y_tr, y_te,
        skip_rbf_tfidf=True,
    )

    print_predictions_summary(all_results, y_te)

    print("📦  Results dict keys:")
    for feat, models in all_results.items():
        print(f"   [{feat}]  →  {list(models.keys())}")
