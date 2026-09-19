"""
reviewer_analyses.py
Les trois analyses demandées par la relecture du benchmark.

    python src/reviewer_analyses.py                  # tout
    python src/reviewer_analyses.py --steps ladder
    python src/reviewer_analyses.py --quick

POURQUOI CES TROIS ANALYSES
---------------------------
A. ESCALIER DE VARIABLES.
   Le gain de 0,152 à 0,185 du schéma riche mélange deux natures de variables :
   l'évaluation clinique courante (type de douleur thoracique, ECG de repos) et
   l'épreuve d'effort (fréquence cardiaque maximale, angine d'effort, sous-décalage
   ST), cette dernière étant un examen réalisé pour investiguer la maladie même
   qu'on prédit. L'escalier à trois marches sépare les deux contributions. Si le
   gain se concentre sur les variables d'effort, on le dit ; s'il est réparti, le
   message informationnel en sort renforcé. Dans les deux cas le résultat est
   plus défendable qu'une mise en garde qualitative.

B. SANS AUCUN RÉGLAGE D'HYPERPARAMÈTRES.
   La sélection d'hyperparamètres n'était pas imbriquée, et rien ne garantit que
   l'optimisme soit identique entre algorithmes : il croît avec la dimension de
   l'espace de recherche, donc il gonfle davantage XGBoost et LightGBM que la
   régression logistique. Le biais joue donc CONTRE la conclusion de l'article.
   Refaire le classement sans aucun réglage teste cela directement : si la
   régression logistique tient toujours son rang, l'objection tombe.

C. VALIDATION CROISÉE IMBRIQUÉE.
   Sur les cohortes de taille modeste, une double boucle est abordable. Elle donne
   des estimations sans optimisme de sélection et vérifie que les conclusions
   comparatives survivent. Kaggle est exclue par défaut (coût), et l'option
   --with-kaggle permet de l'inclure si le temps le permet.

ENTRÉES : data/harmonized/, data/harmonized_rich/
SORTIES : results/tables/rev2_*.csv, results/figures/rev2_*.png
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class CappedSVC(BaseEstimator, ClassifierMixin):
    """SVM RBF entraîné sur un sous-échantillon stratifié d'au plus `cap` patients.

    Le noyau RBF coûte O(n²) et `probability=True` déclenche une validation croisée
    interne, ce qui rend un ajustement direct impraticable au-delà de quelques
    dizaines de milliers de patients. Le plafond reproduit exactement le protocole
    de l'analyse principale ; la prédiction porte toujours sur toutes les données.
    """

    def __init__(self, C=1.0, gamma="scale", cap=8000, class_weight="balanced", random_state=42):
        self.C = C; self.gamma = gamma; self.cap = cap
        self.class_weight = class_weight; self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X); y = np.asarray(y)
        if len(y) > self.cap:
            rng = np.random.default_rng(self.random_state)
            idx = []
            for cls in np.unique(y):
                w = np.where(y == cls)[0]
                k = max(1, int(round(self.cap * len(w) / len(y))))
                idx.append(rng.choice(w, min(k, len(w)), replace=False))
            idx = np.concatenate(idx); X, y = X[idx], y[idx]
        self.classes_ = np.unique(y)
        self._m = SVC(C=self.C, gamma=self.gamma, probability=True,
                      class_weight=self.class_weight, random_state=self.random_state).fit(X, y)
        return self

    def predict_proba(self, X):
        return self._m.predict_proba(np.asarray(X))

    def predict(self, X):
        return self._m.predict(np.asarray(X))

warnings.filterwarnings("ignore")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
THIN = ROOT / "data" / "harmonized"
RICH = ROOT / "data" / "harmonized_rich"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
for d in (TABLES, FIGURES):
    d.mkdir(parents=True, exist_ok=True)
TARGET = "target"
UCI = ["cleveland", "hungarian", "va_longbeach"]
N_BOOT = 1000

# --- les trois marches de l'escalier ---------------------------------------
COMMON5 = ["age", "sex", "sbp", "chol_mgdl", "fbs"]      # présentes partout
CLINICAL = ["cp", "restecg"]                              # évaluation courante
EXERCISE = ["thalach", "exang", "oldpeak"]                # épreuve d'effort
ALIAS = {"chol_mgdl": ["chol_mgdl", "chol"], "fbs": ["fbs", "diabetes"]}


def resolve(cols, wanted):
    """Retrouve les colonnes malgré les variantes de nommage."""
    out = []
    for w in wanted:
        for cand in ALIAS.get(w, [w]):
            if cand in cols:
                out.append(cand); break
    return out


def ci(fn, n, seed, *arrays):
    rng = np.random.default_rng(seed); vals = []
    m = len(arrays[0])
    for _ in range(n):
        i = rng.integers(0, m, m)
        try:
            v = fn(*[a[i] for a in arrays])
            if v == v: vals.append(v)
        except Exception:
            pass
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


# =============================================================== modèles
def default_models(seed=SEED):
    """Hyperparamètres par défaut : aucun réglage, aucun optimisme de sélection."""
    m = {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "nb": GaussianNB(),
        "dtree": DecisionTreeClassifier(class_weight="balanced", random_state=seed),
        "knn": KNeighborsClassifier(),
        "svm": CappedSVC(random_state=seed),
        "rf": RandomForestClassifier(n_jobs=4, class_weight="balanced_subsample", random_state=seed),
        "extra": ExtraTreesClassifier(n_jobs=4, class_weight="balanced", random_state=seed),
        "hgb": HistGradientBoostingClassifier(random_state=seed),
    }
    if HAS_XGB: m["xgb"] = XGBClassifier(eval_metric="logloss", random_state=seed, verbosity=0)
    if HAS_LGB: m["lgbm"] = LGBMClassifier(random_state=seed, verbose=-1, class_weight="balanced")
    return m


def pipe(est):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", est)])


SPACES = {
    "logreg": lambda t: dict(C=t.suggest_float("C", 1e-3, 1e2, log=True)),
    "nb": lambda t: dict(var_smoothing=t.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)),
    "dtree": lambda t: dict(max_depth=t.suggest_int("max_depth", 2, 20),
                            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 40)),
    "knn": lambda t: dict(n_neighbors=t.suggest_int("n_neighbors", 3, 40),
                          weights=t.suggest_categorical("weights", ["uniform", "distance"])),
    "svm": lambda t: dict(C=t.suggest_float("C", 1e-2, 1e2, log=True),
                          gamma=t.suggest_float("gamma", 1e-4, 1.0, log=True)),  # via CappedSVC
    "rf": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 400),
                         max_depth=t.suggest_int("max_depth", 3, 20),
                         min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20)),
    "extra": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 400),
                            max_depth=t.suggest_int("max_depth", 3, 20),
                            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 20)),
    "hgb": lambda t: dict(max_iter=t.suggest_int("max_iter", 100, 500),
                          learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                          max_leaf_nodes=t.suggest_int("max_leaf_nodes", 15, 63)),
    "xgb": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 600),
                          learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                          max_depth=t.suggest_int("max_depth", 3, 10)),
    "lgbm": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 100, 600),
                           learning_rate=t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                           max_depth=t.suggest_int("max_depth", 3, 10)),
}


def load(folder):
    out = {}
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f)
        out[f.stem] = (d[[c for c in d.columns if c != TARGET]].reset_index(drop=True),
                       d[TARGET].values.astype(int))
    return out


def oof(est, X, y, seed=SEED, folds=5):
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    return cross_val_predict(pipe(clone(est)), X, y, cv=cv, method="predict_proba")[:, 1]


# =============================================================== A. escalier
def step_ladder():
    """Sépare l'apport de l'évaluation clinique de celui de l'épreuve d'effort."""
    print("\n=== A. Escalier de variables : commun -> clinique -> épreuve d'effort ===")
    rich = load(RICH)
    if not rich:
        print(f"  [skip] aucune cohorte dans {RICH}"); return pd.DataFrame()
    models = default_models()
    rows = []
    for cohort, (X, y) in rich.items():
        cols = list(X.columns)
        c5 = resolve(cols, COMMON5)
        cl = resolve(cols, CLINICAL)
        ex = resolve(cols, EXERCISE)
        steps = {"1. common five": c5,
                 "2. + clinical assessment": c5 + cl,
                 "3. + exercise test": c5 + cl + ex}
        print(f"\n  {cohort} (n = {len(y)})")
        for name, feats in steps.items():
            missing = [f for f in feats if f not in cols]
            if missing:
                print(f"    [warn] {name}: colonnes absentes {missing}"); continue
            preds = {k: oof(m, X[feats], y) for k, m in models.items()}
            mean_p = np.mean(list(preds.values()), axis=0)
            a = roc_auc_score(y, mean_p)
            best = max(preds, key=lambda k: roc_auc_score(y, preds[k]))
            rows.append({"cohort": cohort, "step": name, "n_features": len(feats),
                         "features": "|".join(feats),
                         "auroc_mean_prediction": round(a, 4),
                         "auroc_best_model": round(roc_auc_score(y, preds[best]), 4),
                         "best_model": best,
                         "auroc_logreg": round(roc_auc_score(y, preds["logreg"]), 4)})
            print(f"    {name:26s} {len(feats):2d} var  AUROC {a:.4f}")
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.to_csv(TABLES / "rev2_A_feature_ladder.csv", index=False)

    print("\n  Gains par marche (prédiction moyenne) :")
    for cohort, g in df.groupby("cohort"):
        g = g.sort_values("n_features")
        v = g.auroc_mean_prediction.values
        if len(v) == 3:
            print(f"    {cohort:14s} clinique {v[1]-v[0]:+.4f}   effort {v[2]-v[1]:+.4f}   "
                  f"total {v[2]-v[0]:+.4f}   part de l'effort {100*(v[2]-v[1])/max(v[2]-v[0],1e-9):.0f} %")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for cohort, g in df.groupby("cohort"):
        g = g.sort_values("n_features")
        ax.plot(range(len(g)), g.auroc_mean_prediction, marker="o", lw=1.8, label=cohort)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Five common\nvariables", "+ clinical\nassessment", "+ exercise\ntest"], fontsize=9)
    ax.set_ylabel("AUROC (mean prediction across learners)"); ax.grid(alpha=.25)
    ax.set_title("Where the gain from the richer schema comes from")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(FIGURES / "rev2_A_feature_ladder.png", dpi=300); plt.close(fig)
    return df


# =============================================================== B. sans réglage
def step_defaults(only=None):
    """Classement des algorithmes sans aucun réglage d'hyperparamètres."""
    print("\n=== B. Classement sans réglage d'hyperparamètres ===")
    data = load(THIN)
    if only:
        data = {k: v for k, v in data.items() if k in only}
        print(f"  cohortes retenues : {list(data)}")
    models = default_models()
    rows = []
    for cohort, (X, y) in data.items():
        print(f"\n  {cohort} (n = {len(y)})")
        preds = {}
        for k, m in models.items():
            try:
                preds[k] = oof(m, X, y)
            except Exception as e:
                print(f"    [warn] {k}: {e}")
        base = preds.get("logreg")
        for k, p in preds.items():
            a = roc_auc_score(y, p)
            d = a - roc_auc_score(y, base) if base is not None else np.nan
            rows.append({"cohort": cohort, "algo": k, "auroc": round(a, 4),
                         "delta_vs_logreg": round(d, 4)})
        best = max((k for k in preds if k != "logreg"), key=lambda k: roc_auc_score(y, preds[k]))
        lo, hi = ci(lambda yy, pb, pl: roc_auc_score(yy, pb) - roc_auc_score(yy, pl),
                    N_BOOT, SEED, y, preds[best], base)
        d = roc_auc_score(y, preds[best]) - roc_auc_score(y, base)
        print(f"    meilleur concurrent : {best}  Δ = {d:+.4f}  IC [{lo:+.4f}, {hi:+.4f}]")
        rows.append({"cohort": cohort, "algo": f"BEST({best}) vs logreg", "auroc": np.nan,
                     "delta_vs_logreg": round(d, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
    df = pd.DataFrame(rows)
    out = TABLES / ("rev2_B_no_tuning.csv" if not only else
                f"rev2_B_no_tuning_{'_'.join(sorted(only))}.csv")
    df.to_csv(out, index=False)
    r = df[df.algo.str.startswith("BEST")]
    print(f"\n  Sans réglage : gain moyen {r.delta_vs_logreg.mean():+.4f}, "
          f"maximum {r.delta_vs_logreg.max():+.4f}")
    print("  (à comparer aux +0,004 moyen et +0,010 maximum obtenus avec 50 essais)")
    return df


# =============================================================== C. CV imbriquée
def step_nested(trials=25, with_kaggle=False):
    """Double boucle : réglage strictement dans le pli d'entraînement."""
    print("\n=== C. Validation croisée imbriquée ===")
    if not HAS_OPTUNA:
        print("  [skip] optuna absent"); return pd.DataFrame()
    data = load(THIN)
    if not with_kaggle:
        data = {k: v for k, v in data.items() if "kaggle" not in k}
        print("  cohorte Kaggle exclue (utiliser --with-kaggle pour l'inclure)")
    base = default_models()
    rows = []
    for cohort, (X, y) in data.items():
        print(f"\n  {cohort} (n = {len(y)})")
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for algo, est in base.items():
            if algo not in SPACES: continue
            pred = np.zeros(len(y))
            for tr, te in outer.split(X, y):
                Xtr, ytr = X.iloc[tr], y[tr]
                def obj(t):
                    p = SPACES[algo](t)
                    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
                    s = cross_val_predict(pipe(clone(est).set_params(**p)), Xtr, ytr,
                                          cv=inner, method="predict_proba")[:, 1]
                    return roc_auc_score(ytr, s)
                st = optuna.create_study(direction="maximize",
                                         sampler=optuna.samplers.TPESampler(seed=SEED))
                st.optimize(obj, n_trials=trials, show_progress_bar=False)
                mdl = pipe(clone(est).set_params(**st.best_params)).fit(Xtr, ytr)
                pred[te] = mdl.predict_proba(X.iloc[te])[:, 1]
            rows.append({"cohort": cohort, "algo": algo, "auroc_nested": round(roc_auc_score(y, pred), 4)})
            print(f"    {algo:7s} AUROC {rows[-1]['auroc_nested']:.4f}")
    df = pd.DataFrame(rows)
    if df.empty: return df
    df.to_csv(TABLES / "rev2_C_nested_cv.csv", index=False)
    print("\n  Meilleur concurrent contre la régression logistique, par cohorte :")
    for cohort, g in df.groupby("cohort"):
        lr = g[g.algo == "logreg"].auroc_nested.iloc[0]
        o = g[g.algo != "logreg"]
        b = o.loc[o.auroc_nested.idxmax()]
        print(f"    {cohort:14s} logreg {lr:.4f}   {b.algo} {b.auroc_nested:.4f}   Δ {b.auroc_nested-lr:+.4f}")
    return df


# =============================================================== C-bis. IC imbriqué
def step_nested_ci(cohort="cleveland", algos=("logreg", "dtree"), trials=25):
    """Validation croisée imbriquée sur UNE cohorte et quelques algorithmes,
    en conservant les prédictions pour calculer un intervalle de confiance.

    L'étape `nested` ne garde que les AUROC, ce qui interdit tout intervalle sur
    la différence entre deux algorithmes. Ici les prédictions hors-échantillon
    sont sauvegardées, ce qui permet un bootstrap apparié sur les mêmes patients.
    """
    print(f"\n=== C-bis. Intervalle de confiance imbriqué — {cohort} ===")
    if not HAS_OPTUNA:
        print("  [skip] optuna absent"); return
    data = load(THIN)
    if cohort not in data:
        print(f"  [skip] cohorte {cohort} absente ({list(data)})"); return
    X, y = data[cohort]
    base = default_models()
    preds = {}
    outdir = ROOT / "results" / "models_nested"
    outdir.mkdir(parents=True, exist_ok=True)
    for algo in algos:
        if algo not in base or algo not in SPACES:
            print(f"  [skip] {algo} indisponible"); continue
        est = base[algo]
        pred = np.zeros(len(y))
        outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for tr, te in outer.split(X, y):
            Xtr, ytr = X.iloc[tr], y[tr]
            def obj(t):
                pr = SPACES[algo](t)
                inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
                sc = cross_val_predict(pipe(clone(est).set_params(**pr)), Xtr, ytr,
                                       cv=inner, method="predict_proba")[:, 1]
                return roc_auc_score(ytr, sc)
            st = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
            st.optimize(obj, n_trials=trials, show_progress_bar=False)
            mdl = pipe(clone(est).set_params(**st.best_params)).fit(Xtr, ytr)
            pred[te] = mdl.predict_proba(X.iloc[te])[:, 1]
        preds[algo] = pred
        np.save(outdir / f"nested_{cohort}_{algo}.npy", pred)
        print(f"  {algo:7s} AUROC {roc_auc_score(y, pred):.4f}")

    rows = []
    ref = algos[0]
    if ref not in preds:
        print("  [skip] référence absente"); return
    for algo in preds:
        if algo == ref: continue
        d = roc_auc_score(y, preds[algo]) - roc_auc_score(y, preds[ref])
        lo, hi = ci(lambda yy, pa, pb: roc_auc_score(yy, pa) - roc_auc_score(yy, pb),
                    N_BOOT, SEED, y, preds[algo], preds[ref])
        rows.append({"cohort": cohort, "algo": algo, "reference": ref,
                     "delta_auroc": round(d, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)})
        print(f"  {algo} − {ref} : {d:+.4f}  IC 95% [{lo:+.4f}, {hi:+.4f}]  "
              f"{'exclut zéro' if (lo > 0 or hi < 0) else 'contient zéro'}")
    if rows:
        pd.DataFrame(rows).to_csv(TABLES / f"rev2_Cbis_nested_ci_{cohort}.csv", index=False)


# =============================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="all", help="all | ladder | defaults | nested | nestedci")
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--with-kaggle", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cohort", default="cleveland",
                    help="cohorte unique pour nestedci")
    ap.add_argument("--algos", default="logreg,dtree",
                    help="algorithmes pour nestedci, le premier sert de référence")
    ap.add_argument("--cohorts", default=None,
                    help="liste séparée par des virgules, ex. kaggle_cvd70k")
    a = ap.parse_args()
    if a.quick: a.trials = 10
    if a.steps in ("all", "ladder"): step_ladder()
    only = [c.strip() for c in a.cohorts.split(",")] if a.cohorts else None
    if a.steps in ("all", "defaults"): step_defaults(only)
    if a.steps in ("all", "nested"): step_nested(a.trials, a.with_kaggle)
    if a.steps == "nestedci":
        step_nested_ci(a.cohort, tuple(x.strip() for x in a.algos.split(",")), a.trials)
    print(f"\nTerminé. Tableaux dans {TABLES}, figure dans {FIGURES}.")


if __name__ == "__main__":
    main()
