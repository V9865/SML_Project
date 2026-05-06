"""
AG News Multi-Class Text Classification — Dimensionality Reduction
==================================================================
Applies Truncated SVD (on TF-IDF) and PCA (on embeddings) across
multiple target dimensions, then retrains all classifiers on the
reduced representations and compares performance.

Why reduce dimensions?
----------------------
  TF-IDF matrices are very high-dimensional (20,000+ features) but
  most variance is concentrated in a much lower-dimensional subspace.
  Reducing dimensions:
    • Removes noise / redundant features → less overfitting
    • Speeds up training (especially KNN, SVM RBF, KMeans)
    • Enables 2D/3D visualisation of feature space
    • For embeddings (768-dim BERT): PCA removes correlated directions
      that do not contribute to class separation

  Trade-off: information loss at very low dim values may hurt accuracy.
  The right dimensionality is found empirically via the explained
  variance curve — look for the "elbow" where marginal gain flattens.

Modules
-------
  TruncatedSVD  — works on sparse TF-IDF (no centering needed)
  PCA           — works on dense BERT embeddings (centers the data)

Integration
-----------
  from ag_news_preprocessing  import main as preprocess
  from ag_news_features       import extract_features
  from ag_news_embeddings     import get_embeddings
  from ag_news_dim_reduction  import run_dim_reduction

  X_train, X_test, y_train, y_test, vocab = preprocess()
  _, _, X_tfidf_tr, X_tfidf_te, _, _ = extract_features(X_train, X_test)
  emb = get_embeddings(X_train, X_test)
  X_bert_tr = emb['train']['cls']   # (6400, 768)
  X_bert_te = emb['test']['cls']    # (1600, 768)

  results = run_dim_reduction(
      X_tfidf_tr, X_tfidf_te,
      X_bert_tr,  X_bert_te,
      y_train,    y_test,
  )
"""

# ── Imports ────────────────────────────────────────────────────────────────────
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend → saves to file
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

from sklearn.decomposition    import TruncatedSVD, PCA
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC, SVC
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.cluster          import KMeans
from sklearn.preprocessing    import Normalizer
from sklearn.metrics          import accuracy_score, f1_score
from sklearn.model_selection  import train_test_split

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
N_CLASSES      = 4
DIM_VALUES     = [50, 100, 200]     # n_components to try
PLOT_DIR       = "."                # where to save figures
KNN_NEIGHBORS  = 7

AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _to_dense(X):
    """Sparse → dense ndarray if needed."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


def _kmeans_map(cluster_ids: np.ndarray, true_labels: np.ndarray,
                n_clusters: int) -> np.ndarray:
    """Map KMeans cluster indices → class labels via majority voting."""
    mapping = {}
    for k in range(n_clusters):
        mask = cluster_ids == k
        mapping[k] = (
            Counter(true_labels[mask]).most_common(1)[0][0]
            if mask.sum() > 0 else 0
        )
    return np.array([mapping[c] for c in cluster_ids])


def _evaluate(y_true, y_pred) -> dict:
    """Return accuracy and macro F1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. DIMENSIONALITY REDUCTION — TRUNCATED SVD  (TF-IDF / sparse)
# ─────────────────────────────────────────────────────────────────────────────

def apply_svd(X_train_sparse, X_test_sparse, n_components: int):
    """
    Truncated SVD — Latent Semantic Analysis (LSA) for sparse matrices
    -------------------------------------------------------------------
    Works directly on sparse TF-IDF without mean-centering the data,
    which would destroy sparsity (and requires dense memory).

    Mathematically:
        X ≈ U  Σ  Vᵀ
        reduced = X · Vᵀ[:n_components]  →  (n_docs, n_components)

    This is equivalent to PCA on centred data when applied to dense arrays,
    but TruncatedSVD is the standard choice for NLP sparse matrices.

    Returns
    -------
    X_tr_svd, X_te_svd  : dense arrays of shape (n, n_components)
    svd                 : fitted TruncatedSVD object
    explained_var_ratio : cumulative explained variance at this n_components
    """
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    X_tr_svd = svd.fit_transform(X_train_sparse)
    X_te_svd = svd.transform(X_test_sparse)

    cum_var = svd.explained_variance_ratio_.sum()
    return X_tr_svd, X_te_svd, svd, cum_var


def svd_explained_variance_curve(X_train_sparse, max_components: int = 300):
    """
    Fit TruncatedSVD at max_components and return the cumulative explained
    variance array (one value per component).  Used to draw the elbow plot.
    """
    n_max = min(max_components, X_train_sparse.shape[1] - 1,
                X_train_sparse.shape[0] - 1)
    svd   = TruncatedSVD(n_components=n_max, random_state=RANDOM_STATE)
    svd.fit(X_train_sparse)
    return np.cumsum(svd.explained_variance_ratio_), n_max


# ─────────────────────────────────────────────────────────────────────────────
# 2. DIMENSIONALITY REDUCTION — PCA  (BERT embeddings / dense)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pca(X_train_dense, X_test_dense, n_components: int):
    """
    PCA on dense BERT embeddings
    ----------------------------
    PCA centres the data (subtracts the mean), computes the covariance
    matrix, then projects onto the top-k eigenvectors (principal components)
    that capture the most variance.

    For BERT CLS / mean-pool vectors (768-dim), PCA:
      • removes correlated dimensions that redundantly encode the same info
      • makes downstream classifiers faster (768 → 50-200 dims)
      • often improves KNN and KMeans accuracy by removing noisy dims

    Returns
    -------
    X_tr_pca, X_te_pca : arrays of shape (n, n_components)
    pca                : fitted PCA object
    explained_var_ratio: cumulative explained variance at this n_components
    """
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_tr_pca = pca.fit_transform(X_train_dense)
    X_te_pca = pca.transform(X_test_dense)

    cum_var = pca.explained_variance_ratio_.sum()
    return X_tr_pca, X_te_pca, pca, cum_var


def pca_explained_variance_curve(X_train_dense, max_components: int = 300):
    """
    Fit PCA at max_components and return cumulative explained variance array.
    """
    n_max = min(max_components, X_train_dense.shape[1],
                X_train_dense.shape[0] - 1)
    pca = PCA(n_components=n_max, random_state=RANDOM_STATE)
    pca.fit(X_train_dense)
    return np.cumsum(pca.explained_variance_ratio_), n_max


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLASSIFIER SUITE  (lean version for reduced feature spaces)
# ─────────────────────────────────────────────────────────────────────────────

def _train_suite(X_tr, X_te, y_tr, y_te, skip_rbf: bool = False) -> dict:
    """
    Train Logistic Regression, Linear SVM, (RBF SVM), KNN, and KMeans
    on the given dense feature matrices.  Returns metrics dict.
    """
    y_tr = np.array(y_tr)
    y_te = np.array(y_te)
    results = {}

    # Logistic Regression
    lr = LogisticRegression(solver="lbfgs", max_iter=2_000,
                            C=1.0, random_state=RANDOM_STATE)
    lr.fit(X_tr, y_tr)
    results["logreg"] = _evaluate(y_te, lr.predict(X_te))

    # Linear SVM
    lsvc = LinearSVC(C=1.0, max_iter=2_000, random_state=RANDOM_STATE)
    lsvc.fit(X_tr, y_tr)
    results["svm_linear"] = _evaluate(y_te, lsvc.predict(X_te))

    # RBF SVM (only if requested and dataset not too large)
    if not skip_rbf:
        norm  = Normalizer(norm="l2")
        X_tr_ = norm.fit_transform(X_tr)
        X_te_ = norm.transform(X_te)
        rsvc  = SVC(kernel="rbf", C=10.0, gamma="scale",
                    decision_function_shape="ovr", random_state=RANDOM_STATE)
        rsvc.fit(X_tr_, y_tr)
        results["svm_rbf"] = _evaluate(y_te, rsvc.predict(X_te_))

    # KNN
    knn = KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS,
                               metric="cosine", algorithm="brute", n_jobs=-1)
    knn.fit(X_tr, y_tr)
    results["knn"] = _evaluate(y_te, knn.predict(X_te))

    # KMeans (unsupervised → majority-vote mapping)
    km  = KMeans(n_clusters=N_CLASSES, init="k-means++", n_init=10,
                 random_state=RANDOM_STATE, max_iter=300)
    km.fit(X_tr)
    preds_km = _kmeans_map(km.predict(X_te), y_te, N_CLASSES)
    results["kmeans"] = _evaluate(y_te, preds_km)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 4. PER-DIM SWEEP — reduce + retrain
# ─────────────────────────────────────────────────────────────────────────────

def sweep_dimensions(
    X_tr_orig, X_te_orig,
    y_train, y_test,
    reducer_fn,          # apply_svd or apply_pca
    dim_values: list     = DIM_VALUES,
    feature_label: str   = "tfidf",
    skip_rbf: bool       = False,
) -> dict:
    """
    For each n_components value in dim_values:
      1. Apply the given reducer to (X_tr_orig, X_te_orig)
      2. Record cumulative explained variance
      3. Retrain all classifiers and record accuracy + macro F1

    Returns
    -------
    sweep_results : {
        n_components: {
            "explained_var": float,
            "X_tr_reduced": ndarray,
            "X_te_reduced": ndarray,
            "metrics": { model_name: {"accuracy": ..., "macro_f1": ...} }
        }
    }
    """
    print(f"\n🔻  Dimension sweep — [{feature_label}]")
    print(f"   Trying n_components = {dim_values}")

    sweep = {}
    for n in dim_values:
        print(f"\n   ── n_components = {n} ──")
        X_tr_r, X_te_r, _, cum_var = reducer_fn(X_tr_orig, X_te_orig, n)
        print(f"      Explained variance : {cum_var*100:.1f}%")
        print(f"      Shape after reduce : {X_tr_r.shape}", end="  →  ")

        metrics = _train_suite(X_tr_r, X_te_r, y_train, y_test,
                               skip_rbf=skip_rbf)

        # Pretty-print accuracy row
        row = "  ".join(
            f"{m}:{v['accuracy']*100:.1f}%" for m, v in metrics.items()
        )
        print(row)

        sweep[n] = {
            "explained_var": cum_var,
            "X_tr_reduced":  X_tr_r,
            "X_te_reduced":  X_te_r,
            "metrics":       metrics,
        }

    return sweep


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "logreg":     "#4C72B0",
    "svm_linear": "#DD8452",
    "svm_rbf":    "#55A868",
    "knn":        "#C44E52",
    "kmeans":     "#8172B2",
}

MODEL_LABELS = {
    "logreg":     "Logistic Reg.",
    "svm_linear": "SVM (Linear)",
    "svm_rbf":    "SVM (RBF)",
    "knn":        "KNN",
    "kmeans":     "KMeans",
}


def _elbow_subplot(ax, cum_var_arr, dim_values, title, color):
    """Draw cumulative explained variance curve with markers for chosen dims."""
    x = np.arange(1, len(cum_var_arr) + 1)
    ax.plot(x, cum_var_arr * 100, color=color, linewidth=2.0, label="Cum. Var.")
    ax.fill_between(x, cum_var_arr * 100, alpha=0.12, color=color)

    for d in dim_values:
        if d <= len(cum_var_arr):
            v = cum_var_arr[d - 1] * 100
            ax.axvline(d, color="grey", linewidth=0.9, linestyle="--", alpha=0.7)
            ax.scatter([d], [v], color=color, s=70, zorder=5)
            ax.annotate(f"{d}d\n{v:.0f}%", (d, v),
                        textcoords="offset points", xytext=(6, -12),
                        fontsize=7.5, color="dimgrey")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("Number of components", fontsize=9)
    ax.set_ylabel("Cumulative explained variance (%)", fontsize=9)
    ax.set_ylim(0, 102)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def _perf_subplot(ax, sweep_results, metric_key, title, marker="o"):
    """Draw accuracy / F1 vs n_components for all models in the sweep."""
    dims = sorted(sweep_results.keys())
    models = list(next(iter(sweep_results.values()))["metrics"].keys())

    for model in models:
        vals = [sweep_results[d]["metrics"][model][metric_key] for d in dims]
        ax.plot(dims, [v * 100 for v in vals],
                marker=marker, linewidth=1.8, markersize=6,
                label=MODEL_LABELS.get(model, model),
                color=PALETTE.get(model, None))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("n_components", fontsize=9)
    ax.set_ylabel(f"{metric_key.replace('_', ' ').title()} (%)", fontsize=9)
    ax.set_xticks(dims)
    ax.set_ylim(0, 102)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def _bar_subplot(ax, sweep_results, metric_key, title):
    """Bar chart: side-by-side models for each n_components."""
    dims   = sorted(sweep_results.keys())
    models = list(next(iter(sweep_results.values()))["metrics"].keys())
    x      = np.arange(len(dims))
    width  = 0.8 / len(models)

    for i, model in enumerate(models):
        vals = [sweep_results[d]["metrics"][model][metric_key] * 100
                for d in dims]
        offset = (i - len(models) / 2) * width + width / 2
        bars = ax.bar(x + offset, vals, width,
                      label=MODEL_LABELS.get(model, model),
                      color=PALETTE.get(model, None), alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("n_components", fontsize=9)
    ax.set_ylabel(f"{metric_key.replace('_', ' ').title()} (%)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=7.5, framealpha=0.7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def plot_dim_reduction_results(
    svd_curve,   svd_sweep,
    pca_curve,   pca_sweep,
    dim_values:  list   = DIM_VALUES,
    save_path:   str    = "dim_reduction_results.png",
):
    """
    Generate a 3×2 figure:
      Row 1 — Explained variance elbow (SVD | PCA)
      Row 2 — Accuracy vs n_components (SVD | PCA)
      Row 3 — Macro F1 bar chart (SVD | PCA)
    """
    fig = plt.figure(figsize=(14, 15))
    fig.suptitle(
        "Dimensionality Reduction — Truncated SVD (TF-IDF) vs PCA (BERT Embeddings)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    # ── Row 1 : Elbow plots ───────────────────────────────────────────────────
    ax00 = fig.add_subplot(gs[0, 0])
    ax01 = fig.add_subplot(gs[0, 1])
    _elbow_subplot(ax00, svd_curve, dim_values,
                   "Truncated SVD — Cumulative Explained Variance (TF-IDF)",
                   "#4C72B0")
    _elbow_subplot(ax01, pca_curve, dim_values,
                   "PCA — Cumulative Explained Variance (BERT Embeddings)",
                   "#C44E52")

    # ── Row 2 : Accuracy line plots ───────────────────────────────────────────
    ax10 = fig.add_subplot(gs[1, 0])
    ax11 = fig.add_subplot(gs[1, 1])
    _perf_subplot(ax10, svd_sweep, "accuracy",
                  "Accuracy vs n_components — SVD (TF-IDF)")
    _perf_subplot(ax11, pca_sweep, "accuracy",
                  "Accuracy vs n_components — PCA (Embeddings)")

    # ── Row 3 : Macro F1 bar charts ───────────────────────────────────────────
    ax20 = fig.add_subplot(gs[2, 0])
    ax21 = fig.add_subplot(gs[2, 1])
    _bar_subplot(ax20, svd_sweep, "macro_f1",
                 "Macro F1 Score — SVD (TF-IDF)")
    _bar_subplot(ax21, pca_sweep, "macro_f1",
                 "Macro F1 Score — PCA (Embeddings)")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n💾  Plot saved → {save_path}")
    return save_path


def plot_2d_scatter(
    X_tr_reduced, y_train,
    title: str = "2D Projection",
    save_path: str = "scatter_2d.png",
):
    """
    Project the first 2 principal components and colour by class label.
    Useful for visualising cluster separability after reduction.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    y = np.array(y_train)

    for cls_id in sorted(np.unique(y)):
        mask = y == cls_id
        ax.scatter(
            X_tr_reduced[mask, 0], X_tr_reduced[mask, 1],
            c=colors[cls_id % len(colors)],
            label=AG_NEWS_LABELS.get(cls_id, str(cls_id)),
            alpha=0.45, s=12, linewidths=0,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Component 1", fontsize=9)
    ax.set_ylabel("Component 2", fontsize=9)
    ax.legend(fontsize=9, framealpha=0.8, markerscale=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"💾  Scatter saved → {save_path}")
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(svd_sweep: dict, pca_sweep: dict):
    """Print a compact accuracy table: dim × model × feature_type."""
    dims   = sorted(svd_sweep.keys())
    models = list(next(iter(svd_sweep.values()))["metrics"].keys())
    label  = {
        "logreg": "LogReg", "svm_linear": "SVM-L",
        "svm_rbf": "SVM-R",  "knn": "KNN", "kmeans": "KMeans",
    }

    print("\n" + "="*78)
    print("  DIMENSIONALITY REDUCTION — ACCURACY SUMMARY")
    print("="*78)
    hdr_models = "  ".join(f"{label.get(m, m):>8}" for m in models)
    print(f"  {'Space':<7}  {'n_dims':>6}  {'ExpVar':>7}  {hdr_models}")
    print("  " + "─"*74)

    for d in dims:
        # SVD row
        ev_svd = svd_sweep[d]["explained_var"]
        svd_acc = "  ".join(
            f"{svd_sweep[d]['metrics'][m]['accuracy']*100:>7.1f}%"
            for m in models
        )
        print(f"  {'SVD':<7}  {d:>6}  {ev_svd*100:>6.1f}%  {svd_acc}")

        # PCA row
        ev_pca = pca_sweep[d]["explained_var"]
        pca_acc = "  ".join(
            f"{pca_sweep[d]['metrics'][m]['accuracy']*100:>7.1f}%"
            for m in models
        )
        print(f"  {'PCA':<7}  {d:>6}  {ev_pca*100:>6.1f}%  {pca_acc}")
        print("  " + "─"*74)

    print("="*78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def run_dim_reduction(
    X_tfidf_train,
    X_tfidf_test,
    X_bert_train,
    X_bert_test,
    y_train,
    y_test,
    dim_values:   list = DIM_VALUES,
    plot_dir:     str  = PLOT_DIR,
    skip_rbf:     bool = False,
) -> dict:
    """
    Full dimensionality reduction pipeline:
      1. Compute explained variance curves (SVD + PCA)
      2. Sweep over dim_values — reduce + retrain
      3. Generate and save all plots
      4. Print summary table

    Parameters
    ----------
    X_tfidf_train / X_tfidf_test  : sparse TF-IDF matrices
    X_bert_train  / X_bert_test   : dense BERT embedding arrays (n, 768)
    y_train / y_test              : integer label arrays
    dim_values                    : list of n_components to try
    plot_dir                      : directory to save figures
    skip_rbf                      : if True, skip RBF SVM (saves time)

    Returns
    -------
    {
        "svd_sweep": dict,
        "pca_sweep": dict,
        "svd_curve": ndarray,
        "pca_curve": ndarray,
        "plot_path": str,
        "scatter_svd": str,
        "scatter_pca": str,
    }
    """
    os.makedirs(plot_dir, exist_ok=True)
    X_bert_tr = _to_dense(X_bert_train)
    X_bert_te = _to_dense(X_bert_test)

    # ── Explained variance curves ─────────────────────────────────────────────
    print("\n📈  Computing explained variance curves …")
    max_c = max(dim_values) + 50          # a bit beyond the highest target dim
    svd_curve, _ = svd_explained_variance_curve(X_tfidf_train, max_c)
    pca_curve, _ = pca_explained_variance_curve(X_bert_tr,     max_c)

    # ── Dimension sweeps ──────────────────────────────────────────────────────
    svd_sweep = sweep_dimensions(
        X_tfidf_train, X_tfidf_test,
        y_train, y_test,
        reducer_fn    = apply_svd,
        dim_values    = dim_values,
        feature_label = "TF-IDF → SVD",
        skip_rbf      = True,     # linear SVM dominates on TF-IDF anyway
    )

    pca_sweep = sweep_dimensions(
        X_bert_tr, X_bert_te,
        y_train, y_test,
        reducer_fn    = apply_pca,
        dim_values    = dim_values,
        feature_label = "BERT → PCA",
        skip_rbf      = skip_rbf,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n🎨  Generating plots …")
    plot_path = plot_dim_reduction_results(
        svd_curve, svd_sweep,
        pca_curve, pca_sweep,
        dim_values = dim_values,
        save_path  = os.path.join(plot_dir, "dim_reduction_results.png"),
    )

    # 2-D scatter for the best dim (100 components)
    best_dim = 100 if 100 in dim_values else dim_values[-1]
    scatter_svd = plot_2d_scatter(
        svd_sweep[best_dim]["X_tr_reduced"], y_train,
        title     = f"SVD 2D Projection — TF-IDF (n={best_dim})",
        save_path = os.path.join(plot_dir, "scatter_svd_2d.png"),
    )
    scatter_pca = plot_2d_scatter(
        pca_sweep[best_dim]["X_tr_reduced"], y_train,
        title     = f"PCA 2D Projection — BERT Embeddings (n={best_dim})",
        save_path = os.path.join(plot_dir, "scatter_pca_2d.png"),
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary_table(svd_sweep, pca_sweep)

    return {
        "svd_sweep":   svd_sweep,
        "pca_sweep":   pca_sweep,
        "svd_curve":   svd_curve,
        "pca_curve":   pca_curve,
        "plot_path":   plot_path,
        "scatter_svd": scatter_svd,
        "scatter_pca": scatter_pca,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. STANDALONE DEMO  (synthetic data — matches real pipeline shapes exactly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scipy.sparse import random as sp_random

    print("🧪  Standalone demo with synthetic data")
    print("    (Replace with real pipeline arrays on your machine)\n")

    np.random.seed(RANDOM_STATE)
    N_TRAIN, N_TEST = 800, 200

    # Synthetic sparse TF-IDF (800 docs × 2000 features, ~5% dense)
    X_tfidf_tr = sp_random(N_TRAIN, 2000, density=0.05, format="csr",
                           random_state=RANDOM_STATE)
    X_tfidf_te = sp_random(N_TEST,  2000, density=0.05, format="csr",
                           random_state=RANDOM_STATE + 1)

    # Synthetic dense BERT embeddings (800 docs × 768 dims)
    # Inject class signal so PCA has something to separate
    y_tr = np.array([i % N_CLASSES for i in range(N_TRAIN)])
    y_te = np.array([i % N_CLASSES for i in range(N_TEST)])

    class_means = np.random.randn(N_CLASSES, 768) * 5
    X_bert_tr = np.vstack([
        class_means[y_tr[i]] + np.random.randn(768) for i in range(N_TRAIN)
    ])
    X_bert_te = np.vstack([
        class_means[y_te[i]] + np.random.randn(768) for i in range(N_TEST)
    ])

    # ── Run the full pipeline ─────────────────────────────────────────────────
    results = run_dim_reduction(
        X_tfidf_tr, X_tfidf_te,
        X_bert_tr,  X_bert_te,
        y_tr, y_te,
        dim_values = [50, 100, 200],
        plot_dir   = ".",
        skip_rbf   = True,     # speeds up demo; set False for full run
    )

    print("\n📦  Return dict keys:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"   {k} → {list(v.keys())}")
        elif isinstance(v, np.ndarray):
            print(f"   {k} → ndarray shape {v.shape}")
        else:
            print(f"   {k} → {v}")

    print("\n✅  Dimensionality reduction demo complete.")
    print("    On your machine, plug in real TF-IDF + BERT arrays from the pipeline.")
