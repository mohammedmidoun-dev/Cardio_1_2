"""
robustness.py
Five sensitivity analyses answering the objections a reviewer will raise.

    python src/robustness.py                 # all five analyses
    python src/robustness.py --quick         # smaller bootstrap, faster

WHAT IT ANSWERS
---------------
A1. FEATURE-AVAILABILITY CONFOUND (the objection that can sink the paper).
    Cleveland and Statlog observe 5 variables, Framingham and Kaggle 8. The
    transfer penalty may therefore reflect a degenerate feature space rather
    than a divergent risk structure — and the SHAP/transfer correlation may be
    produced by the same common factor. Everything is recomputed on the FIVE
    variables shared by all four cohorts.

A2. NON-INDEPENDENT PAIRS. The six cohort pairs share cohorts, so a Pearson
    p-value is anticonservative. Replaced by an EXACT Mantel permutation test
    over all 4! = 24 relabellings (minimum attainable p = 1/24 = 0.042).

A3. MIXED MODEL FAMILIES IN THE SHAP ANALYSIS. The published SHAP rankings come
    from XGBoost, CatBoost and LightGBM depending on cohort, so cohort effects
    and model effects are entangled. Recomputed with ONE family everywhere.

A4. KAGGLE OUTLIERS. Physiologically impossible values (e.g. mean diastolic
    pressure of 96.6 mmHg) are filtered and the analysis repeated.

A5. NO UNCERTAINTY ON THE TRANSFER PENALTY. Bootstrap confidence intervals for
    every transfer AUROC and for the 0.112 penalty.

INPUT  : data/harmonized/{cohort}.csv  (produced by harmonize.py)
OUTPUT : results/tables/robustness_*.csv|json, results/figures/figR1_*.png
"""
from __future__ import annotations
import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
COHORTS = ["uci_cleveland", "framingham", "kaggle_cvd70k", "statlog"]
FEATURES8 = ["age", "sex", "sbp", "dbp", "chol", "smoke", "diabetes", "bmi"]
FEATURES5 = ["age", "sex", "sbp", "chol", "diabetes"]   # observed in all four
TARGET = "target"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "harmonized"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Physiological bounds for A4 (Kaggle cleaning)
BOUNDS = {"sbp": (70, 250), "dbp": (40, 150), "bmi": (12, 60), "age": (18, 100)}


# --------------------------------------------------------------- data
def load(cohort: str, features, clean_kaggle: bool = False) -> tuple[pd.DataFrame, np.ndarray]:
    path = DATA / f"{cohort}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if clean_kaggle and cohort == "kaggle_cvd70k":
        n0 = len(df)
        m = pd.Series(True, index=df.index)
        for c, (lo, hi) in BOUNDS.items():
            if c in df.columns:
                m &= df[c].between(lo, hi) | df[c].isna()
        if {"sbp", "dbp"}.issubset(df.columns):
            m &= df["sbp"] > df["dbp"]
        df = df.loc[m]
        print(f"    [A4] {cohort}: {n0} -> {len(df)} patients "
              f"({100*(n0-len(df))/n0:.1f}% removed as physiologically implausible)")
    df = df.reset_index(drop=True)          # labels must match positions in y
    return df[features].copy(), df[TARGET].values.astype(int)


def make_pipe(kind: str) -> Pipeline:
    """Fixed, defensible hyperparameters: this is a sensitivity analysis, so
    tuning is deliberately held constant across cohorts and schemas."""
    if kind == "logreg":
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 random_state=SEED)
    elif kind == "dtree":
        clf = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                     class_weight="balanced", random_state=SEED)
    elif kind == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                     min_samples_leaf=5, n_jobs=6,
                                     class_weight="balanced_subsample",
                                     random_state=SEED)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=250, max_depth=6,
                                             learning_rate=0.06,
                                             random_state=SEED)
    else:
        raise ValueError(kind)
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", clf)])


PANEL = ["logreg", "dtree", "rf", "hgb"]


# --------------------------------------------------------------- A1 + A5
def transfer_matrix(features, clean_kaggle=False, n_boot=200, seed=SEED):
    """Full transfer matrix averaged over the panel, with bootstrap CIs."""
    data = {}
    for c in COHORTS:
        try:
            data[c] = load(c, features, clean_kaggle)
        except FileNotFoundError:
            print(f"    [skip] {c}: harmonised file missing")
    coh = list(data)
    fitted = {}
    for c in coh:
        X, y = data[c]
        fitted[c] = {k: make_pipe(k).fit(X, y) for k in PANEL}

    rng = np.random.default_rng(seed)
    rows, mat = [], pd.DataFrame(index=coh, columns=coh, dtype=float)
    for src in coh:
        for tgt in coh:
            X, y = data[tgt]
            aucs = []
            for k in PANEL:
                if src == tgt:                      # honest within-cohort: OOF
                    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
                    p = cross_val_predict(make_pipe(k), X, y, cv=cv,
                                          method="predict_proba")[:, 1]
                else:
                    p = fitted[src][k].predict_proba(X)[:, 1]
                aucs.append(roc_auc_score(y, p))
                if k == "rf":                        # bootstrap on one model
                    bs = []
                    for _ in range(n_boot):
                        idx = rng.integers(0, len(y), len(y))
                        if len(np.unique(y[idx])) < 2:
                            continue
                        bs.append(roc_auc_score(y[idx], p[idx]))
                    lo, hi = np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan)
            mat.loc[src, tgt] = float(np.mean(aucs))
            rows.append({"train": src, "test": tgt,
                         "auroc_mean_panel": round(float(np.mean(aucs)), 4),
                         "auroc_rf": round(float(aucs[PANEL.index("rf")]), 4),
                         "rf_ci_low": round(float(lo), 4), "rf_ci_high": round(float(hi), 4)})
    return mat.astype(float), pd.DataFrame(rows), data


def penalty_with_ci(mat: pd.DataFrame, detail: pd.DataFrame):
    v = mat.values
    diag = np.diag(v)
    off = v.copy(); np.fill_diagonal(off, np.nan)
    pen = float(np.nanmean(diag) - np.nanmean(off))
    # CI by resampling the off-diagonal cell CIs
    lo = float(np.nanmean(diag) - np.nanmean(detail.loc[detail.train != detail.test, "rf_ci_high"]))
    hi = float(np.nanmean(diag) - np.nanmean(detail.loc[detail.train != detail.test, "rf_ci_low"]))
    return {"within_mean": round(float(np.nanmean(diag)), 4),
            "transfer_mean": round(float(np.nanmean(off)), 4),
            "penalty": round(pen, 4),
            "penalty_ci_approx": [round(min(lo, hi), 4), round(max(lo, hi), 4)]}


# --------------------------------------------------------------- A3
def shap_rankings(data, features, seed=SEED, n_sample=2000):
    """One model family everywhere (random forest) to remove the model effect."""
    out = {}
    rng = np.random.default_rng(seed)
    for c, (X, y) in data.items():
        pipe = make_pipe("rf").fit(X, y)
        pos = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
        Xs, ys = X.iloc[pos], y[pos]        # positional: robust to filtered rows
        Xt = pipe[:-1].transform(Xs)
        try:
            import shap
            v = shap.TreeExplainer(pipe[-1]).shap_values(Xt)
            if isinstance(v, list):
                v = v[1]
            v = np.asarray(v)
            if v.ndim == 3:
                v = v[:, :, -1]
            imp = pd.Series(np.abs(v).mean(0), index=features)
        except Exception as e:                       # fallback: permutation
            print(f"    [note] SHAP unavailable ({e}); using permutation importance")
            from sklearn.inspection import permutation_importance
            r = permutation_importance(pipe, Xs, ys, n_repeats=5,
                                       random_state=seed, scoring="roc_auc")
            imp = pd.Series(r.importances_mean, index=features)
        out[c] = (imp / imp.sum()).sort_values(ascending=False)
    return out


def tau_loss_table(rank: dict, mat: pd.DataFrame):
    rows = []
    for a, b in itertools.combinations(list(rank), 2):
        ra, rb = rank[a], rank[b]
        common = ra.index.intersection(rb.index)
        tau, _ = kendalltau(ra[common].rank(ascending=False),
                            rb[common].rank(ascending=False))
        loss = np.mean([mat.loc[b, b] - mat.loc[a, b], mat.loc[a, a] - mat.loc[b, a]])
        rows.append({"pair": f"{a} <-> {b}", "a": a, "b": b,
                     "kendall_tau": round(float(tau), 3),
                     "transfer_mean": round(float((mat.loc[a, b] + mat.loc[b, a]) / 2), 3),
                     "transfer_loss": round(float(loss), 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- A2
def mantel_exact(rank: dict, mat: pd.DataFrame):
    """Exact Mantel test over all 4! = 24 relabellings of the cohorts.

    D1 = 1 - tau  (dissimilarity of SHAP profiles)
    D2 = transfer loss (symmetrised)
    Minimum attainable p is 1/24 = 0.042 with four cohorts.
    """
    coh = list(rank)
    n = len(coh)
    D1 = np.zeros((n, n)); D2 = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        a, b = coh[i], coh[j]
        ra, rb = rank[a], rank[b]
        common = ra.index.intersection(rb.index)
        tau, _ = kendalltau(ra[common].rank(ascending=False),
                            rb[common].rank(ascending=False))
        loss = np.mean([mat.loc[b, b] - mat.loc[a, b], mat.loc[a, a] - mat.loc[b, a]])
        D1[i, j] = D1[j, i] = 1 - tau
        D2[i, j] = D2[j, i] = loss

    def stat(perm):
        P1 = D1[np.ix_(perm, perm)]
        iu = np.triu_indices(n, 1)
        return pearsonr(P1[iu], D2[iu])[0]

    obs = stat(list(range(n)))
    perms = list(itertools.permutations(range(n)))
    null = np.array([stat(list(p)) for p in perms])
    p_val = float(np.mean(null >= obs))
    return {"mantel_r_observed": round(float(obs), 3),
            "n_permutations": len(perms),
            "p_exact_one_sided": round(p_val, 4),
            "min_attainable_p": round(1 / len(perms), 4),
            "note": "D1 = 1 - Kendall tau between SHAP rankings; D2 = transfer loss. "
                    "With four cohorts only 24 relabellings exist, so p cannot fall "
                    "below 0.042 whatever the effect size."}


# --------------------------------------------------------------- driver
def run_block(label, features, clean_kaggle, n_boot):
    print(f"\n=== {label} ===")
    mat, detail, data = transfer_matrix(features, clean_kaggle, n_boot)
    pen = penalty_with_ci(mat, detail)
    print(mat.round(3).to_string())
    print(f"  within {pen['within_mean']}  transfer {pen['transfer_mean']}  "
          f"penalty {pen['penalty']} (approx. 95% CI {pen['penalty_ci_approx']})")
    rank = shap_rankings(data, features)
    tl = tau_loss_table(rank, mat)
    r, p = pearsonr(tl.kendall_tau, tl.transfer_loss)
    rs, ps = spearmanr(tl.kendall_tau, tl.transfer_loss)
    mant = mantel_exact(rank, mat)
    print(tl.to_string(index=False))
    print(f"  Pearson r = {r:.3f} (p = {p:.4f}) | Spearman rho = {rs:.3f}")
    print(f"  Exact Mantel: r = {mant['mantel_r_observed']}, "
          f"p = {mant['p_exact_one_sided']} (floor {mant['min_attainable_p']})")
    return {"matrix": mat, "detail": detail, "penalty": pen, "rank": rank,
            "tau_loss": tl, "pearson_r": round(float(r), 3), "pearson_p": round(float(p), 4),
            "spearman_rho": round(float(rs), 3), "mantel": mant}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer bootstrap resamples")
    a = ap.parse_args()
    n_boot = 50 if a.quick else 200

    blocks = {
        "A1_five_common_features": run_block(
            "A1 — restricted to the five variables observed in ALL cohorts",
            FEATURES5, False, n_boot),
        "reference_eight_features": run_block(
            "Reference — eight-variable schema (as in the manuscript)",
            FEATURES8, False, n_boot),
        "A4_kaggle_cleaned": run_block(
            "A4 — five variables, Kaggle cohort filtered to physiological ranges",
            FEATURES5, True, n_boot),
    }

    for name, b in blocks.items():
        b["matrix"].round(4).to_csv(TABLES / f"robustness_{name}_transfer.csv")
        b["detail"].to_csv(TABLES / f"robustness_{name}_transfer_ci.csv", index=False)
        b["tau_loss"].to_csv(TABLES / f"robustness_{name}_tau_loss.csv", index=False)
        pd.DataFrame(b["rank"]).round(4).to_csv(TABLES / f"robustness_{name}_shap.csv")

    summary = {n: {"penalty": b["penalty"], "pearson_r": b["pearson_r"],
                   "pearson_p": b["pearson_p"], "spearman_rho": b["spearman_rho"],
                   "mantel": b["mantel"]} for n, b in blocks.items()}
    with open(TABLES / "robustness_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Figure R1: tau vs transfer loss under each schema
    fig, ax = plt.subplots(figsize=(6.5, 5))
    styles = {"reference_eight_features": ("o", "Eight-variable schema"),
              "A1_five_common_features": ("s", "Five common variables"),
              "A4_kaggle_cleaned": ("^", "Five variables, Kaggle cleaned")}
    for name, (mk, lab) in styles.items():
        d = blocks[name]["tau_loss"]
        ax.scatter(d.kendall_tau, d.transfer_loss, marker=mk, s=55,
                   label=f"{lab} (r = {blocks[name]['pearson_r']:.2f})")
        if len(d) > 2:
            z = np.polyfit(d.kendall_tau, d.transfer_loss, 1)
            xs = np.linspace(d.kendall_tau.min(), d.kendall_tau.max(), 20)
            ax.plot(xs, np.polyval(z, xs), lw=1, alpha=.6)
    ax.set_xlabel("Kendall τ between cohort SHAP rankings")
    ax.set_ylabel("Mean transfer loss (AUROC)")
    ax.set_title("Explanation agreement versus transfer loss")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "figR1_tau_vs_loss_sensitivity.png", dpi=300)
    plt.close(fig)

    print("\n================ VERDICT ================")
    ref = blocks["reference_eight_features"]; a1 = blocks["A1_five_common_features"]
    print(f"  8 variables : r = {ref['pearson_r']}, penalty = {ref['penalty']['penalty']}")
    print(f"  5 variables : r = {a1['pearson_r']}, penalty = {a1['penalty']['penalty']}")
    if a1["pearson_r"] <= -0.6:
        print("  -> The association SURVIVES restriction to a common feature space.")
        print("     The feature-availability confound does not explain it; report A1")
        print("     as the primary sensitivity analysis in the revised manuscript.")
    else:
        print("  -> The association WEAKENS once the feature space is equalised.")
        print("     Report this honestly: the manuscript's claim must be softened to")
        print("     an association observed under unequal feature availability.")
    print(f"\n  Files written to {TABLES} and {FIGURES}")


if __name__ == "__main__":
    main()
