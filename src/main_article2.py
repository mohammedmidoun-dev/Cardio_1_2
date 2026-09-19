"""
main_article2.py
Orchestrator for Article 2: 12 src->tgt pairs x 7 methods x 3 label fractions
x 5 seeds = up to 1260 experimental cells (some invalid combinations are
skipped).

Reuses everything from Article 1:
  - data/harmonized/<cohort>.csv
  - results/models/<cohort>_<algo>_best.joblib
  - config.py, pipeline.py, models.py, stats.py

Run from the project root:
    python src/main_article2.py
"""
from __future__ import annotations
import time
import warnings
from itertools import product

import joblib
import numpy as np
import pandas as pd
from optuna.trial import FixedTrial

from config import COHORTS, MODELS_DIR, TABLES_DIR
from pipeline import load_harmonized, evaluate, make_full_pipeline
from models import FACTORY
from adaptation import ADAPTERS
from shift_diagnosis import full_shift_report
from federated import fed_avg, fed_predict_proba

warnings.filterwarnings("ignore")

BASE_ALGO = "xgb"                       # winner from Article 1 (adjust if needed)
LABEL_FRACTIONS = [0.0, 0.10, 0.50]
SEEDS = [0, 1, 2, 3, 4]

# Methods that need labelled target data (frac > 0)
NEEDS_LABELS = {"platt", "isotonic", "tradaboost", "finetune"}


# --------------------------------------------------------- Target splits
def split_target(X, y, frac_labeled: float, seed: int):
    """30% held-out test; the remainder split into (labelled, unlabelled)
    pools according to `frac_labeled` (relative to the full target)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(0.30 * len(X))
    test = idx[:n_test]
    rest = idx[n_test:]
    n_lab = int(frac_labeled * len(X))
    lab, unlab = rest[:n_lab], rest[n_lab:]
    return X[lab], y[lab], X[unlab], X[test], y[test]


# --------------------------------------------------------- One experiment
def _base_factory(src_cohort: str):
    """Return a callable that produces a fresh pipeline cloned from the
    Article-1 best model for `src_cohort` (with its tuned hyperparameters)."""
    base = joblib.load(MODELS_DIR / f"{src_cohort}_{BASE_ALGO}_best.joblib")
    params = base.named_steps["clf"].get_params()
    def factory():
        trial = FixedTrial(params)
        estimator = FACTORY[BASE_ALGO](trial)
        return make_full_pipeline(BASE_ALGO, estimator, sampler=None)
    return factory


def run_one(src: str, tgt: str, method: str, frac: float, seed: int) -> dict:
    base = joblib.load(MODELS_DIR / f"{src}_{BASE_ALGO}_best.joblib")
    Xs, ys, _ = load_harmonized(src)
    Xt, yt, _ = load_harmonized(tgt)
    X_lab, y_lab, X_unlab, X_te, y_te = split_target(Xt, yt, frac, seed)

    if method == "baseline":
        prob = base.predict_proba(X_te)[:, 1]
    else:
        AdapterCls = ADAPTERS[method]
        # Adapters that need a factory (refit / continued training)
        if method in ("iw", "coral"):
            adapter = AdapterCls(_base_factory(src))
        else:
            adapter = AdapterCls()

        adapter.fit(
            base, Xs, ys,
            X_target_unlab=X_unlab if len(X_unlab) > 0 else None,
            X_target_lab=X_lab if frac > 0 else None,
            y_target_lab=y_lab if frac > 0 else None,
        )
        prob = adapter.predict_proba(X_te)[:, 1]

    m = evaluate(y_te, prob)
    m.update({"src": src, "tgt": tgt, "method": method,
              "frac": frac, "seed": seed})
    return m


# --------------------------------------------------------- Shift diagnosis
def step_diagnosis():
    print("\n=== Step 1/3 — Shift diagnosis ===")
    rows_summary = []
    rows_ks = []
    for src, tgt in product(COHORTS, COHORTS):
        if src == tgt:
            continue
        Xs, ys, names = load_harmonized(src)
        Xt, yt, _     = load_harmonized(tgt)
        rep = full_shift_report(Xs, ys, Xt, yt, names)
        rows_summary.append({
            "src": src, "tgt": tgt,
            "mmd":        rep["mmd"],
            "domain_auc": rep["domain_auc"],
            **rep["prevalence"],
        })
        ks = rep["ks"].copy()
        ks["src"] = src; ks["tgt"] = tgt
        rows_ks.append(ks)
    pd.DataFrame(rows_summary).to_csv(
        TABLES_DIR / "article2_shift_summary.csv", index=False)
    pd.concat(rows_ks, ignore_index=True).to_csv(
        TABLES_DIR / "article2_shift_ks_per_feature.csv", index=False)
    print(f"  -> {TABLES_DIR/'article2_shift_summary.csv'}")


# --------------------------------------------------------- Main loop
def step_adaptation_loop():
    print("\n=== Step 2/3 — Adaptation experiments ===")
    rows = []
    methods = ["baseline"] + list(ADAPTERS.keys())
    pairs = [(s, t) for s, t in product(COHORTS, COHORTS) if s != t]
    total = len(pairs) * len(methods) * len(LABEL_FRACTIONS) * len(SEEDS)
    print(f"  estimated cells: {total}")

    t0 = time.time()
    for src, tgt in pairs:
        for method in methods:
            for frac in LABEL_FRACTIONS:
                # Skip invalid combinations
                if method in NEEDS_LABELS and frac == 0.0:
                    continue
                if method == "baseline" and frac > 0.0:
                    # baseline doesn't depend on target labels; record once
                    continue
                for seed in SEEDS:
                    try:
                        rows.append(run_one(src, tgt, method, frac, seed))
                    except Exception as e:
                        print(f"  [warn] {src}->{tgt}/{method}/frac={frac}/"
                              f"seed={seed}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "article2_adaptation_results.csv", index=False)
    print(f"  saved {len(df)} cells -> "
          f"{TABLES_DIR/'article2_adaptation_results.csv'}")
    print(f"  elapsed: {(time.time()-t0)/60:.1f} min")


# --------------------------------------------------------- Federated upper bound
def step_federated():
    print("\n=== Step 3/3 — Federated upper bound (FedAvg) ===")
    client_data = []
    for c in COHORTS:
        X, y, _ = load_harmonized(c)
        # Impute NaNs with median so SGD doesn't choke (handled in pipeline
        # for the other methods but FedAvg works on raw arrays here)
        med = np.nanmedian(X, axis=0)
        idx = np.where(np.isnan(X))
        X[idx] = np.take(med, idx[1])
        client_data.append((X, y))

    rows = []
    for tgt_idx, tgt in enumerate(COHORTS):
        # Hold out 30% of the target cohort as eval; train federated on the
        # other three cohorts plus the 70% remaining of the target.
        rng = np.random.default_rng(0)
        X_tgt, y_tgt = client_data[tgt_idx]
        idx = rng.permutation(len(X_tgt))
        n_te = int(0.30 * len(X_tgt))
        te, tr = idx[:n_te], idx[n_te:]
        held = [(X_tgt[tr], y_tgt[tr])] + [
            client_data[j] for j in range(len(COHORTS)) if j != tgt_idx
        ]
        scalers, w, b = fed_avg(held, n_rounds=5, n_local_epochs=3, lr=0.01)
        prob = fed_predict_proba(X_tgt[te], scalers[0], w, b)[:, 1]
        m = evaluate(y_tgt[te], prob)
        m.update({"target": tgt, "method": "fed_avg"})
        rows.append(m)
        print(f"  target={tgt:<15} AUROC={m['auroc']:.3f}  Brier={m['brier']:.3f}")
    pd.DataFrame(rows).to_csv(
        TABLES_DIR / "article2_fed_results.csv", index=False)


# --------------------------------------------------------- Driver
def main():
    step_diagnosis()
    step_adaptation_loop()
    step_federated()
    print("\nDone. All Article-2 outputs are in results/tables/.")


if __name__ == "__main__":
    main()
