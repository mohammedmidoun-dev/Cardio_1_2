"""
figures_v2.py
Figures de publication et métriques de seuil, à partir des modèles entraînés
par analysis_v2.py (results/models_v2). Aucun réentraînement d'hyperparamètres.

    python src/figures_v2.py --schema L1            # analyse principale
    python src/figures_v2.py --schema L2            # schéma riche (cohortes UCI)
    python src/figures_v2.py --schema L1 --features common5
    python src/figures_v2.py --quick                # bootstrap réduit

POURQUOI UN NOUVEAU FICHIER
---------------------------
figures.py lisait results/models/{cohort}_{algo}_best.joblib, convention de la
version 1. analysis_v2.py écrit results/models_v2/{tag}_{cohort}_{algo}.joblib
et, surtout, sauvegarde les prédictions HORS-ÉCHANTILLON (oof_*.npy). Ce script
réutilise directement ces prédictions : les courbes, les seuils et les matrices
de confusion sont donc cohérents avec les AUROC rapportées, sans le moindre
recalcul optimiste en échantillon.

SORTIES
-------
figures/  figV2_{tag}_roc_{cohorte}.png          courbes ROC
          figV2_{tag}_calibration_{cohorte}.png  calibration + ECE
          figV2_{tag}_dca_{cohorte}.png          courbes de décision
          figV2_{tag}_confusion_{cohorte}.png    matrices de confusion (Youden)
          figV2_{tag}_auroc_box.png              AUROC par algorithme
          figV2_{tag}_transfer_heatmap.png       matrice de transfert
          figV2_{tag}_effect_forest.png          tailles d'effet vs logreg
          figV2_levers.png                       les trois leviers comparés
tables/   v2_{tag}_threshold_metrics.csv         Se, Sp, VPP, VPN, accuracy...
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DIRS = {"L1": ROOT / "data" / "harmonized", "L2": ROOT / "data" / "harmonized_rich"}
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models_v2"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "target"
COMMON5 = ["age", "sex", "sbp", "chol", "diabetes"]
DPI = 300
ECE_BINS = 10
DCA_T = np.linspace(0.01, 0.60, 60)
LABELS = {"logreg": "Logistic regression", "nb": "Naive Bayes", "dtree": "Decision tree",
          "knn": "k-NN", "svm": "SVM (RBF)", "rf": "Random forest",
          "extra": "Extra trees", "hgb": "Hist. gradient boosting",
          "xgb": "XGBoost", "lgbm": "LightGBM"}


# ----------------------------------------------------------------- utilitaires
def ece(y, p, bins=ECE_BINS):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() / len(y)
                     for b in range(bins) if (idx == b).any()))


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def net_benefit(y, p, thresholds=DCA_T):
    y, p = np.asarray(y), np.asarray(p)
    N, pos = len(y), y.sum()
    nb, nb_all = [], []
    for t in thresholds:
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        w = t / (1 - t)
        nb.append(tp / N - (fp / N) * w)
        nb_all.append(pos / N - ((N - pos) / N) * w)
    return np.array(nb), np.array(nb_all)


def load_data(schema, features_mode):
    out = {}
    for f in sorted(DIRS[schema].glob("*.csv")):
        df = pd.read_csv(f)
        feats = [c for c in df.columns if c != TARGET]
        if features_mode == "common5":
            feats = [c for c in feats if c in COMMON5]
        elif features_mode == "nochol":
            feats = [c for c in feats if c not in ("chol", "chol_mgdl")]
        out[f.stem] = (df[feats], df[TARGET].values.astype(int))
    return out


def load_oof(tag, cohort, algos):
    """Prédictions hors-échantillon produites par analysis_v2.py."""
    out = {}
    for a in algos:
        f = MODELS / f"oof_{tag}_{cohort}_{a}.npy"
        if f.exists():
            out[a] = np.load(f)
    return out


# ----------------------------------------------------------------- figures
def fig_roc(tag, cohort, y, preds):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for a, p in sorted(preds.items(), key=lambda kv: -roc_auc_score(y, kv[1])):
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, lw=1.4, label=f"{LABELS.get(a, a)} ({roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance (0.500)")
    ax.set_xlabel("1 - Specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.set_title(f"Out-of-fold ROC curves — {cohort}")
    ax.legend(fontsize=7.5, loc="lower right", title="AUROC")
    ax.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / f"figV2_{tag}_roc_{cohort}.png", dpi=DPI); plt.close(fig)


def fig_calibration(tag, cohort, y, preds):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for a, p in sorted(preds.items(), key=lambda kv: ece(y, kv[1])):
        try:
            frac, mean_p = calibration_curve(y, p, n_bins=ECE_BINS, strategy="quantile")
            ax.plot(mean_p, frac, marker="o", ms=3, lw=1.3,
                    label=f"{LABELS.get(a, a)} (ECE={ece(y, p):.3f})")
        except Exception:
            continue
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(f"Calibration curves — {cohort}")
    ax.legend(fontsize=7.5, loc="upper left"); ax.grid(alpha=.25)
    plt.tight_layout()
    plt.savefig(FIGURES / f"figV2_{tag}_calibration_{cohort}.png", dpi=DPI); plt.close(fig)


def fig_dca(tag, cohort, y, preds, top_k=5):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    best = sorted(preds.items(), key=lambda kv: -roc_auc_score(y, kv[1]))[:top_k]
    nb_all = None
    for a, p in best:
        nb, nb_all = net_benefit(y, p)
        ax.plot(DCA_T, nb, lw=1.5, label=LABELS.get(a, a))
    if nb_all is not None:
        ax.plot(DCA_T, nb_all, "k-", lw=1, alpha=.6, label="treat all")
        ax.axhline(0, color="k", ls="--", lw=1, label="treat none")
        ax.set_ylim(max(min(0, np.nanmin(nb_all)), -0.10), None)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision curve analysis — {cohort}")
    ax.legend(fontsize=8); ax.grid(alpha=.25); plt.tight_layout()
    plt.savefig(FIGURES / f"figV2_{tag}_dca_{cohort}.png", dpi=DPI); plt.close(fig)


def fig_confusion(tag, cohort, y, preds):
    algos = sorted(preds, key=lambda a: -roc_auc_score(y, preds[a]))
    n = len(algos); ncol = min(5, n); nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.5 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, a in zip(axes, algos):
        p = preds[a]; t = youden(y, p)
        cm = confusion_matrix(y, (p >= t).astype(int))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8,
                    color="white" if v > cm.max() / 2 else "black")
        ax.set_title(f"{LABELS.get(a, a)}\nthreshold={t:.2f}", fontsize=8)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["pred. 0", "pred. 1"], fontsize=7)
        ax.set_yticklabels(["true 0", "true 1"], fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices at the Youden threshold — {cohort}", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURES / f"figV2_{tag}_confusion_{cohort}.png", dpi=DPI); plt.close(fig)


def fig_auroc_box(tag):
    f = TABLES / f"v2_{tag}_within.csv"
    if not f.exists():
        return
    df = pd.read_csv(f).drop_duplicates(["cohort", "algo"], keep="last")
    order = df.groupby("algo")["auroc"].median().sort_values(ascending=False).index.tolist()
    data = [df.loc[df.algo == a, "auroc"].values for a in order]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(data, labels=[LABELS.get(a, a) for a in order],
                    patch_artist=True, widths=.6)
    for b in bp["boxes"]:
        b.set_facecolor("#cfe2f3"); b.set_edgecolor("#2E75B6")
    for i, a in enumerate(order, start=1):
        v = df.loc[df.algo == a, "auroc"].values
        ax.scatter(np.full(len(v), i), v, s=14, color="#1F4E79", zorder=3)
    ax.set_ylabel("AUROC (out-of-fold, within cohort)")
    ax.set_title("AUROC distribution per algorithm across cohorts")
    ax.grid(alpha=.25, axis="y"); plt.xticks(rotation=30, ha="right")
    plt.tight_layout(); plt.savefig(FIGURES / f"figV2_{tag}_auroc_box.png", dpi=DPI)
    plt.close(fig)


def fig_transfer_heatmap(tag):
    f = TABLES / f"v2_{tag}_transfer_mean.csv"
    if not f.exists():
        return
    m = pd.read_csv(f, index_col=0)
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(m.values.astype(float), cmap="RdYlBu_r", vmin=0.5, vmax=0.9)
    ax.set_xticks(range(len(m.columns))); ax.set_xticklabels(m.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(m.index))); ax.set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if (v < 0.6 or v > 0.85) else "black")
    ax.set_xlabel("Test cohort"); ax.set_ylabel("Train cohort")
    ax.set_title("Mean cross-cohort AUROC across algorithms")
    fig.colorbar(im, ax=ax, label="AUROC")
    plt.tight_layout(); plt.savefig(FIGURES / f"figV2_{tag}_transfer_heatmap.png", dpi=DPI)
    plt.close(fig)


def fig_effect_forest(tag):
    """Tailles d'effet contre la régression logistique, avec IC : la figure
    demandée par les relecteurs à la place du poids mis sur les p-values."""
    f = TABLES / f"v2_{tag}_effect_sizes.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    g = (d.groupby("algo")
           .agg(delta=("delta_auroc_vs_logreg", "mean"),
                lo=("ci_low", "mean"), hi=("ci_high", "mean"))
           .sort_values("delta"))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ypos = np.arange(len(g))
    ax.errorbar(g.delta, ypos, xerr=[g.delta - g.lo, g.hi - g.delta],
                fmt="o", color="#1F4E79", ecolor="#7FA6CC", capsize=3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(ypos); ax.set_yticklabels([LABELS.get(a, a) for a in g.index])
    ax.set_xlabel("ΔAUROC versus penalised logistic regression (mean across cohorts)")
    ax.set_title("Effect sizes with bootstrap confidence intervals")
    ax.grid(alpha=.25, axis="x"); plt.tight_layout()
    plt.savefig(FIGURES / f"figV2_{tag}_effect_forest.png", dpi=DPI); plt.close(fig)


def fig_levers():
    """Les trois leviers sur la même échelle : le message de l'article."""
    items, vals, colors = [], [], []
    f1 = TABLES / "v2_L1_effect_sizes.csv"
    if f1.exists():
        d = pd.read_csv(f1)
        items.append("Best algorithm vs logistic\nregression (cohort mean)"); colors.append("#7FA6CC")
        vals.append(float(d.groupby("cohort")["delta_auroc_vs_logreg"].max().mean()))
    f2 = TABLES / "v2_information_gain.csv"
    if f2.exists():
        d = pd.read_csv(f2)
        items.append("3 extra routine\nvariables (5→8)"); colors.append("#7FA6CC")
        vals.append(float(d.information_gain.mean()))
    w1, w2 = TABLES / "v2_L1_within.csv", TABLES / "v2_L2_within.csv"
    if w1.exists() and w2.exists():
        a = pd.read_csv(w1).drop_duplicates(["cohort", "algo"], keep="last")
        b = pd.read_csv(w2).drop_duplicates(["cohort", "algo"], keep="last")
        common = set(a.cohort) & set(b.cohort)
        if common:
            ga = a[a.cohort.isin(common)].groupby("cohort")["auroc"].mean()
            gb = b[b.cohort.isin(common)].groupby("cohort")["auroc"].mean()
            items.append("5 rich clinical\nvariables (5→10)"); colors.append("#2E75B6")
            vals.append(float((gb - ga).mean()))
    for tag, lab in [("L2", "Cohort shift,\nshared protocol"),
                     ("L1", "Cohort shift,\nthin harmonisation")]:
        f = TABLES / f"v2_{tag}_transfer_mean.csv"
        if f.exists():
            m = pd.read_csv(f, index_col=0).values.astype(float)
            off = m.copy(); np.fill_diagonal(off, np.nan)
            items.append(lab); colors.append("#C00000")
            vals.append(float(np.nanmean(off) - np.nanmean(np.diag(m))))
    if not items:
        return
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.barh(items, vals, color=colors)
    ax.axvline(0, color="k", lw=1)
    span = max(abs(min(vals)), abs(max(vals)))
    ax.set_xlim(min(min(vals) - 0.35 * span, -0.01), max(vals) + 0.30 * span)
    for i, v in enumerate(vals):
        off = 0.02 * span
        ax.text(v + (off if v >= 0 else -off), i, f"{v:+.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel("Change in AUROC")
    ax.set_title("What actually moves performance: data richness and comparability, not algorithm")
    ax.grid(alpha=.25, axis="x"); plt.tight_layout()
    plt.savefig(FIGURES / "figV2_levers.png", dpi=DPI); plt.close(fig)


# ----------------------------------------------------------------- métriques
def threshold_metrics(tag, cohort, y, preds):
    rows = []
    prev = float(np.mean(y))
    for a, p in preds.items():
        for rule, t in (("youden", youden(y, p)), ("0.5", 0.5)):
            pred = (p >= float(t)).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            rows.append({
                "tag": tag, "cohort": cohort, "algo": a, "threshold_rule": rule,
                "threshold": round(float(t), 4),
                "auroc": round(roc_auc_score(y, p), 4),
                "auprc": round(average_precision_score(y, p), 4),
                "brier": round(brier_score_loss(y, p), 4), "ece": round(ece(y, p), 4),
                "sensitivity": round(tp / (tp + fn), 4) if tp + fn else np.nan,
                "specificity": round(tn / (tn + fp), 4) if tn + fp else np.nan,
                "ppv": round(tp / (tp + fp), 4) if tp + fp else np.nan,
                "npv": round(tn / (tn + fn), 4) if tn + fn else np.nan,
                "f1": round(f1_score(y, pred), 4),
                "accuracy": round(accuracy_score(y, pred), 4),
                "balanced_accuracy": round(balanced_accuracy_score(y, pred), 4),
                "constant_classifier_accuracy": round(max(prev, 1 - prev), 4),
                "prevalence": round(prev, 4),
                "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)})
    return rows


# ----------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="L1", choices=["L1", "L2"])
    ap.add_argument("--features", default="all", choices=["all", "common5", "nochol"])
    ap.add_argument("--cohorts", default="")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    tag = a.schema + {"all": "", "common5": "_c5", "nochol": "_nochol"}[a.features]

    data = load_data(a.schema, a.features)
    if a.cohorts:
        keep = [c.strip() for c in a.cohorts.split(",")]
        data = {k: v for k, v in data.items() if k in keep}
    algos = list(LABELS)
    all_rows = []
    print(f"=== figures pour {tag} ({len(data)} cohortes) ===")
    for cohort, (X, y) in data.items():
        preds = load_oof(tag, cohort, algos)
        if not preds:
            print(f"  [skip] {cohort}: aucune prédiction hors-échantillon "
                  f"(lancer analysis_v2.py --steps within)")
            continue
        print(f"  {cohort}: {len(preds)} modèles")
        fig_roc(tag, cohort, y, preds)
        fig_calibration(tag, cohort, y, preds)
        fig_dca(tag, cohort, y, preds)
        fig_confusion(tag, cohort, y, preds)
        all_rows += threshold_metrics(tag, cohort, y, preds)

    if all_rows:
        df = pd.DataFrame(all_rows)
        out = TABLES / f"v2_{tag}_threshold_metrics.csv"
        df.to_csv(out, index=False)
        print(f"\n-> {out}")
        yo = df[df.threshold_rule == "youden"]
        print("\nAperçu (seuil de Youden) :")
        print(yo[["cohort", "algo", "auroc", "sensitivity", "specificity", "ppv",
                  "accuracy", "constant_classifier_accuracy"]].to_string(index=False))
        # le mirage d'accuracy, sur la cohorte de plus faible prévalence
        low = df.loc[df.prevalence.idxmin(), "cohort"]
        h = df[(df.cohort == low) & (df.threshold_rule == "0.5")]
        if len(h):
            print(f"\nMIRAGE D'ACCURACY — {low} (prévalence {h.prevalence.iloc[0]:.3f}) :")
            print(f"  classifieur constant : {h.constant_classifier_accuracy.iloc[0]:.3f}")
            print(h[["algo", "accuracy", "sensitivity"]].round(4).to_string(index=False))

    fig_auroc_box(tag)
    fig_transfer_heatmap(tag)
    fig_effect_forest(tag)
    fig_levers()
    print(f"\nFigures écrites dans {FIGURES}")


if __name__ == "__main__":
    main()
