"""Synthetic-data experiment on sklearn load_digits (stand-in for document character crops).

Baseline : standard augmentation (random shift, small rotation, pixel noise) x10 + MLP.
Improved : class-conditional shrinkage-Gaussian generator in PCA space, samples kept only if
           a logistic regression trained on the real images gives the intended class p >= 0.8, + same MLP.
"""
import json
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from sklearn.covariance import LedoitWolf
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SEEDS = [42, 43, 44]          # mandated seeds for randomized models
REAL_PER_CLASS = 15           # scarce real budget per class
N_SYN_PER_REAL = 10           # synthetic samples per real image (before filtering)
N_PCA = 10                    # PCA dimensions for the Gaussian generator
KEEP_THR = 0.8                # quality-filter threshold on intended-class probability
PIX_MAX = 1.0                 # pixels scaled to [0, 1]
OUT_DIR = "figures"
T0 = time.time()
warnings.filterwarnings("ignore", message="Maximum number of iterations")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def baseline_augment(X, y, rng):
    """Standard augmentation: random shift (+-1 px), rotation (+-10 deg), Gaussian pixel noise."""
    out_X, out_y = [], []
    for x, c in zip(X, y):
        for _ in range(N_SYN_PER_REAL):
            img = x.reshape(8, 8)
            img = shift(img, rng.uniform(-1, 1, size=2), order=1, mode="constant")
            img = rotate(img, rng.uniform(-10, 10), reshape=False, order=1, mode="constant")
            img = img + rng.normal(0, 1.0 / 16, img.shape)   # noise ~1 grey level
            out_X.append(np.clip(img, 0, PIX_MAX).ravel())
            out_y.append(c)
    return np.array(out_X), np.array(out_y)


def fit_generator(X, y):
    """Per-class Gaussian in PCA space; covariance shrunk with Ledoit-Wolf."""
    pca = PCA(n_components=N_PCA, random_state=SEED).fit(X)
    Z = pca.transform(X)
    stats = {}
    for c in np.unique(y):
        Zc = Z[y == c]
        stats[c] = (Zc.mean(axis=0), LedoitWolf().fit(Zc).covariance_)
    return pca, stats


def sample_generator(pca, stats, n_per_class, rng):
    """Draw n_per_class samples for every class, map back to pixel space."""
    Xs, ys = [], []
    for c, (mu, cov) in stats.items():
        Zs = rng.multivariate_normal(mu, cov, size=n_per_class)
        Xs.append(np.clip(pca.inverse_transform(Zs), 0, PIX_MAX))
        ys.append(np.full(n_per_class, c))
    return np.vstack(Xs), np.concatenate(ys)


def quality_filter(Xs, ys, filt):
    """Keep samples whose intended-class probability under the real-data classifier >= KEEP_THR."""
    proba = filt.predict_proba(Xs)
    col = np.searchsorted(filt.classes_, ys)
    keep = proba[np.arange(len(ys)), col] >= KEEP_THR
    return Xs[keep], ys[keep], keep.mean()


def train_eval(X_train, y_train, X_test, y_test, seed):
    """Same MLP and evaluation for both methods; returns (accuracy, macro-F1)."""
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=seed)
    mlp.fit(X_train, y_train)
    pred = mlp.predict(X_test)
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro")


def main():
    digits = load_digits()
    X = digits.data / 16.0            # pixel values 0..16 -> [0, 1]
    y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    log(f"loaded {len(X)} rows; train={len(X_tr)} test={len(X_te)}")

    res = {"base_acc": [], "base_f1": [], "imp_acc": [], "imp_f1": [], "kept": [], "n_gen": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        # Scarce real budget: 15 real training images per class
        idx = np.concatenate([rng.choice(np.where(y_tr == c)[0], REAL_PER_CLASS, replace=False)
                              for c in range(10)])
        Xr, yr = X_tr[idx], y_tr[idx]

        # Baseline: standard augmentation (10x synthetic) + MLP on real + synthetic
        Xb, yb = baseline_augment(Xr, yr, rng)
        a, f = train_eval(np.vstack([Xr, Xb]), np.concatenate([yr, yb]), X_te, y_te, seed)
        res["base_acc"].append(a); res["base_f1"].append(f)
        log(f"seed {seed} baseline: acc={a:.4f} f1={f:.4f} (synthetic={len(Xb)})")

        # Improved: class-conditional Gaussian generator, same synthetic count before filtering
        pca, stats = fit_generator(Xr, yr)
        Xg, yg = sample_generator(pca, stats, REAL_PER_CLASS * N_SYN_PER_REAL, rng)
        filt = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
        filt.fit(Xr, yr)                       # filter classifier sees only real images
        Xk, yk, kept = quality_filter(Xg, yg, filt)
        if len(Xk) == 0:
            log(f"seed {seed}: filter kept no samples")
            sys.exit(1)
        a, f = train_eval(np.vstack([Xr, Xk]), np.concatenate([yr, yk]), X_te, y_te, seed)
        res["imp_acc"].append(a); res["imp_f1"].append(f)
        res["kept"].append(kept); res["n_gen"].append(len(Xg))
        log(f"seed {seed} improved: acc={a:.4f} f1={f:.4f} kept={kept:.3f} ({len(Xk)}/{len(Xg)})")

    vals = {k: np.array(v, dtype=float) for k, v in res.items()}
    if not all(np.isfinite(v).all() for v in vals.values()):
        log("non-finite metrics; aborting")
        sys.exit(1)

    b_mean, i_mean = vals["base_acc"].mean(), vals["imp_acc"].mean()
    metrics = {
        "baseline_score": float(b_mean),
        "improved_score": float(i_mean),
        "improvement": float(i_mean - b_mean),
        "baseline_score_std": float(vals["base_acc"].std()),
        "improved_score_std": float(vals["imp_acc"].std()),
        "baseline_f1": float(vals["base_f1"].mean()),
        "improved_f1": float(vals["imp_f1"].mean()),
        "kept_fraction": float(vals["kept"].mean()),
        "n_generated_per_seed": int(vals["n_gen"].mean()),
    }
    results = {
        "question": "Does a class-conditional, quality-filtered synthetic generator give a downstream "
                    "classifier more test accuracy than standard image augmentation when only a few "
                    "real labeled examples per class are available?",
        "method": "15 real images/class; baseline = 10x shift/rotate/noise augmentation + MLP; "
                  "improved = per-class shrinkage Gaussian in PCA space, filtered by logistic-regression "
                  "confidence >= 0.8, same MLP; seeds 42-44.",
        "dataset": {
            "name": "scikit-learn load_digits",
            "source_url": "https://scikit-learn.org/stable/datasets/toy_dataset.html",
            "n_rows": int(len(X)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "test accuracy",
        "higher_is_better": True,
        "baseline": "Standard augmentation generator (random shifts, small rotations, pixel noise; "
                    "10 synthetic per real image) with an MLPClassifier trained on real + synthetic.",
        "improved": "Class-conditional Gaussian generator in PCA space with Ledoit-Wolf shrinkage, "
                    "same number of samples before filtering, kept only if a logistic regression trained "
                    "on real images assigns the intended class probability >= 0.8; same MLPClassifier.",
        "metrics": metrics,
        "notes": "Digits are a stand-in for document character crops, not the paper's VLM/diffusion "
                 "pipeline. Hyper-parameters were fixed a priori (no tuning, so no test-set selection; "
                 "no CV search was needed). Three seeds (42-44) per the mandated protocol. The filter "
                 "classifier is trained only on the same real images. Std values are population std "
                 "across seeds and are based on very few runs.",
    }
    with open("results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    os.makedirs(OUT_DIR, exist_ok=True) if False else None
    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["baseline", "improved"], [b_mean, i_mean],
           yerr=[vals["base_acc"].std(), vals["imp_acc"].std()], capsize=6, color=["gray", "tab:blue"])
    ax.set_ylabel("test accuracy (mean over seeds)")
    ax.set_title("Digits, 15 real/class")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "accuracy_bar.png"), dpi=120)
    log(f"baseline={b_mean:.4f} improved={i_mean:.4f} improvement={i_mean - b_mean:+.4f}; wrote results.json")


if __name__ == "__main__":
    main()
