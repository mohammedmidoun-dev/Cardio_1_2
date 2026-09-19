"""
interpret.py
SHAP-based global interpretability for the best model of each cohort.

CORRECTIONS APPORTÉES
---------------------
1. BUG principal : `_transform_X` cherchait une étape nommée "preprocess". Si
   le pipeline ne contient pas ce nom exact, la fonction renvoyait X BRUT à
   l'estimateur final, d'où les deux échecs observés :
     - framingham/svm  : "Input X contains NaN"  (imputation court-circuitée)
     - uci_cleveland/lgbm : "number of features in data (8) is not the same as
       it was in training data (5)"  (les colonnes vides dbp/smoke/bmi sont
       supprimées à l'entraînement mais pas ici)
   -> remplacé par `pipe[:-1].transform(X)`, qui applique TOUTES les étapes de
      prétraitement quel que soit leur nom.

2. Noms de variables récupérés via get_feature_names_out() après transformation
   (Cleveland/Statlog n'ont que 5 variables réelles, pas 8) — sinon le graphique
   SHAP affiche des étiquettes fausses, ce qui est pire qu'une erreur.

3. Vitesse : KernelExplainer tournait 18 min à 23 % sur le SVM de Framingham.
   Fond réduit via shap.kmeans, évaluation plafonnée, et possibilité de
   privilégier le meilleur modèle À BASE D'ARBRES (TreeSHAP, quasi instantané).
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
import joblib

from config import COHORTS, MODELS_DIR, FIGURES_DIR, TABLES_DIR, COMMON_FEATURES
from pipeline import load_harmonized

TREE_ALGOS = {"dtree", "rf", "gb", "xgb", "lgbm", "catboost"}

# Budget pour les explainers non-arbres (SVM, kNN, logreg, nb)
KERNEL_BACKGROUND = 50     # taille du fond (résumé par k-means)
KERNEL_N_EVAL = 200        # nombre de patients expliqués


def _final_estimator(pipe):
    """Dernière étape du pipeline, quel que soit son nom."""
    try:
        return pipe[-1]
    except Exception:
        return pipe.named_steps.get("clf", pipe)


def _preprocess(pipe, X):
    """Applique TOUTES les étapes sauf la dernière (imputation, scaling,
    sélection de colonnes...). C'est le correctif du bug principal."""
    try:
        return np.asarray(pipe[:-1].transform(X))
    except Exception:
        pre = pipe.named_steps.get("preprocess")
        return np.asarray(pre.transform(X)) if pre is not None else np.asarray(X)


def _is_generic(names) -> bool:
    """['x0','x1',...] = noms générés faute de noms d'entrée : inutilisables."""
    return all(re.fullmatch(r"x\d+", str(n)) for n in names)


def _feature_names(pipe, n_cols: int, fallback=None):
    """Noms après prétraitement.

    Le pipeline étant ajusté sur un tableau NumPy, get_feature_names_out()
    renvoie 'x0', 'x1'... de la bonne longueur : il faut donc lui fournir
    explicitement les noms d'entrée, sinon les graphiques SHAP portent des
    étiquettes fausses (erreur silencieuse, plus grave qu'un plantage).
    """
    fallback = list(fallback) if fallback is not None else list(COMMON_FEATURES)

    # 1. voie propre : propager les noms d'origine à travers le prétraitement
    try:
        names = list(pipe[:-1].get_feature_names_out(input_features=fallback))
        names = [str(n).split("__")[-1] for n in names]
        if len(names) == n_cols and not _is_generic(names):
            return names
    except Exception:
        pass

    # 2. repli : appliquer les masques de sélection étape par étape
    try:
        names = list(fallback)
        for _, step in pipe[:-1].steps:
            if hasattr(step, "get_support"):
                names = [n for n, k in zip(names, step.get_support()) if k]
            elif hasattr(step, "statistics_"):      # SimpleImputer : colonnes vides
                keep = ~np.isnan(np.asarray(step.statistics_, dtype=float))
                if keep.sum() != len(names) and len(keep) == len(names):
                    names = [n for n, k in zip(names, keep) if k]
        if len(names) == n_cols:
            return names
    except Exception:
        pass

    # 3. derniers recours
    if len(fallback) == n_cols:
        return fallback
    return [f"f{i}" for i in range(n_cols)]


def _positive_class(shap_values):
    """Uniformise les sorties SHAP (liste, ou tableau 3D des versions récentes)."""
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) == 2 else shap_values[0]
    v = np.asarray(shap_values)
    if v.ndim == 3:                      # (n, features, classes)
        return v[:, :, -1]
    return v


def shap_summary(cohort: str, algo: str, max_display: int = 10,
                 save_table: bool = True):
    pipe = joblib.load(MODELS_DIR / f"{cohort}_{algo}_best.joblib")
    X, y, raw_names = load_harmonized(cohort)

    X_pre = _preprocess(pipe, X)                       # <- CORRECTIF
    estimator = _final_estimator(pipe)
    names = _feature_names(pipe, X_pre.shape[1], raw_names)

    if algo in TREE_ALGOS:
        explainer = shap.TreeExplainer(estimator)
        shap_values = _positive_class(explainer.shap_values(X_pre))
        X_plot = X_pre
    else:
        # fond résumé par k-means : bien plus rapide qu'un échantillon brut
        bg = shap.kmeans(X_pre, min(KERNEL_BACKGROUND, len(X_pre)))
        explainer = shap.KernelExplainer(estimator.predict_proba, bg)
        n_eval = min(KERNEL_N_EVAL, len(X_pre))
        idx = np.random.RandomState(0).choice(len(X_pre), n_eval, replace=False)
        X_plot = X_pre[idx]
        shap_values = _positive_class(
            explainer.shap_values(X_plot, nsamples=100, silent=True))

    fig = plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X_plot, feature_names=names,
                      show=False, max_display=max_display)
    plt.tight_layout()
    out = FIGURES_DIR / f"shap_summary_{cohort}_{algo}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  SHAP summary -> {out.name}   ({len(names)} variables réelles)")

    if save_table:
        imp = (pd.Series(np.abs(shap_values).mean(axis=0), index=names)
                 .sort_values(ascending=False))
        t = TABLES_DIR / f"shap_importance_{cohort}_{algo}.csv"
        imp.to_frame("mean_abs_shap").to_csv(t)
        print(f"    classement -> {t.name}: " +
              ", ".join(f"{k}={v:.3f}" for k, v in imp.head(4).items()))
    return imp if save_table else None


def _best_tree_model_per_cohort(within_csv=None):
    """Meilleur modèle À BASE D'ARBRES par cohorte.

    Justification méthodologique : le test de Nemenyi ne sépare aucun
    algorithme, donc restreindre l'analyse SHAP aux modèles à base d'arbres ne
    coûte pas de performance et permet d'utiliser TreeSHAP (exact et rapide)
    plutôt qu'une approximation par KernelSHAP.
    """
    path = within_csv or (TABLES_DIR / "within_cohort_metrics.csv")
    df = pd.read_csv(path)
    df = df[df["algo"].isin(TREE_ALGOS)]
    best = df.sort_values("auroc_mean", ascending=False).groupby("cohort").head(1)
    return dict(zip(best["cohort"], best["algo"]))


def run_for_best(best_per_cohort: dict[str, str], prefer_tree: bool = True):
    """`best_per_cohort` associe cohorte -> algorithme gagnant.

    prefer_tree=True remplace un gagnant non-arbre (SVM, kNN...) par le
    meilleur modèle à base d'arbres de la même cohorte : TreeSHAP est exact et
    quasi instantané là où KernelSHAP prend des dizaines de minutes.
    """
    mapping = dict(best_per_cohort)
    if prefer_tree:
        try:
            trees = _best_tree_model_per_cohort()
            for c, a in list(mapping.items()):
                if a not in TREE_ALGOS and c in trees:
                    print(f"  [info] {c}: {a} -> {trees[c]} (TreeSHAP exact et rapide)")
                    mapping[c] = trees[c]
        except Exception as e:
            print(f"  [warn] sélection arbre impossible ({e}), mapping conservé")

    for cohort, algo in mapping.items():
        try:
            shap_summary(cohort, algo)
        except Exception as e:
            print(f"  [warn] SHAP failed for {cohort}/{algo}: {e}")


if __name__ == "__main__":
    import sys
    mapping = _best_tree_model_per_cohort()
    print("Meilleur modèle à base d'arbres par cohorte :", mapping)
    run_for_best(mapping, prefer_tree=False)
