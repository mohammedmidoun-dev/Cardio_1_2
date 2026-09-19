"""
analysis_v2.py
Analyse unique et complète de l'étude assainie (phase 3 du plan).

    python src/analysis_v2.py --schema L1              # analyse principale
    python src/analysis_v2.py --schema L2              # schéma riche (UCI)
    python src/analysis_v2.py --quick                  # 3 essais, test de plomberie
    python src/analysis_v2.py --steps within,transfer  # sous-ensemble d'étapes
    python src/analysis_v2.py --resume                 # reprend où ça s'est arrêté

CE QUE CE SCRIPT CORRIGE PAR RAPPORT À LA VERSION 1
---------------------------------------------------
1. Budget d'optimisation UNIFORME (50 essais) pour les dix algorithmes.
   Les budgets inégaux (15/20/30) étaient une faiblesse gratuite.
2. Matrices de transfert PAR ALGORITHME, et non seulement leur moyenne.
3. SHAP SOURCE -> CIBLE : le modèle source est expliqué sur ses propres données
   puis sur les données cible. C'est le seul protocole réellement « sans
   étiquettes », et il sépare la divergence de population de la divergence de
   modèle. La version 1 comparait deux modèles différents, ce qui mélangeait
   les deux effets.
4. UNE SEULE famille de modèles pour toutes les analyses SHAP.
5. Test de MANTEL EXACT au lieu d'un p de Pearson, invalide puisque les paires
   de cohortes ne sont pas indépendantes.
6. TAILLES D'EFFET avec intervalles de confiance bootstrap, plutôt que le seul
   test de Friedman-Nemenyi (faible puissance avec peu de cohortes).
7. Courbe d'apprentissage du SVM : justifie empiriquement le sous-échantillonnage
   au lieu de l'excuser par le temps de calcul.
8. Gain informationnel (5 -> 8 variables) mesuré dans la même unité que le gain
   algorithmique et la perte de transfert.
9. Analyse de sensibilité sans le cholestérol.
10. Distribution des probabilités de Naive Bayes par repli.

ENTRÉES : data/harmonized/*.csv (L1), data/harmonized_rich/*.csv (L2)
SORTIES : results/tables/v2_*.csv, results/figures/v2_*.png, v2_summary.json
"""
from __future__ import annotations
import argparse
import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, kendalltau, pearsonr, spearmanr
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DIRS = {"L1": ROOT / "data" / "harmonized", "L2": ROOT / "data" / "harmonized_rich"}
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
MODELS = ROOT / "results" / "models_v2"
for d in (TABLES, FIGURES, MODELS):
    d.mkdir(parents=True, exist_ok=True)

TARGET = "target"
ALGOS = ["logreg", "nb", "dtree", "knn", "svm", "rf", "extra", "hgb", "xgb", "lgbm"]
TREE_ALGOS = {"dtree", "rf", "extra", "hgb", "xgb", "lgbm"}
SHAP_FAMILY = "rf"          # une seule famille pour toutes les analyses SHAP
SVM_MAX_N = 8000
N_BOOT = 500
ECE_BINS = 10

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


# =============================================================== modèles
class FastSVC(__import__("sklearn").base.ClassifierMixin,
              __import__("sklearn").base.BaseEstimator):
    """SVM RBF rendu traitable : sous-échantillon stratifié + Platt sur un
    unique découpage retenu (au lieu d'une validation croisée interne à 5
    replis, soit un coût cinq fois moindre)."""

    def __init__(self, C=1.0, gamma=0.1, max_train_n=SVM_MAX_N, random_state=SEED):
        self.C, self.gamma = C, gamma
        self.max_train_n, self.random_state = max_train_n, random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.max_train_n and len(X) > self.max_train_n:
            X, _, y, _ = train_test_split(X, y, train_size=int(self.max_train_n),
                                          stratify=y, random_state=self.random_state)
        strat = y if np.bincount(y.astype(int)).min() >= 5 else None
        Xf, Xc, yf, yc = train_test_split(X, y, test_size=0.2, stratify=strat,
                                          random_state=self.random_state)
        self.svc_ = SVC(C=self.C, gamma=self.gamma, kernel="rbf", probability=False,
                        cache_size=1000, max_iter=200_000, class_weight="balanced",
                        random_state=self.random_state).fit(Xf, yf)
        d = self.svc_.decision_function(Xc).reshape(-1, 1)
        self.platt_ = (LogisticRegression(max_iter=1000).fit(d, yc)
                       if len(np.unique(yc)) > 1 else None)
        return self

    def predict_proba(self, X):
        d = self.svc_.decision_function(np.asarray(X, dtype=np.float32)).reshape(-1, 1)
        if self.platt_ is not None:
            return self.platt_.predict_proba(d)
        p = 1 / (1 + np.exp(-d.ravel()))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def space(algo, t):
    if algo == "logreg":
        return {"C": t.suggest_float("C", 1e-3, 1e2, log=True),
                "penalty": t.suggest_categorical("penalty", ["l1", "l2"])}
    if algo == "nb":
        return {"var_smoothing": t.suggest_float("var_smoothing", 1e-12, 1e-2, log=True)}
    if algo == "dtree":
        return {"max_depth": t.suggest_int("max_depth", 2, 20),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 40)}
    if algo == "knn":
        return {"n_neighbors": t.suggest_int("n_neighbors", 3, 40),
                "weights": t.suggest_categorical("weights", ["uniform", "distance"])}
    if algo == "svm":
        return {"C": t.suggest_float("C", 1e-2, 1e2, log=True),
                "gamma": t.suggest_float("gamma", 1e-4, 1.0, log=True)}
    if algo in ("rf", "extra"):
        return {"n_estimators": t.suggest_int("n_estimators", 100, 400, step=100),
                "max_depth": t.suggest_int("max_depth", 3, 20),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 1, 20)}
    if algo == "hgb":
        return {"max_iter": t.suggest_int("max_iter", 100, 500, step=50),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 63)}
    if algo in ("xgb", "lgbm"):
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600, step=100),
                "learning_rate": t.suggest_float("learning_rate", 1e-2, 0.3, log=True),
                "max_depth": t.suggest_int("max_depth", 3, 10)}
    raise ValueError(algo)


def estimator(algo, p, n_jobs=4):
    if algo == "logreg":
        return LogisticRegression(C=p["C"], penalty=p["penalty"], solver="liblinear",
                                  class_weight="balanced", max_iter=3000, random_state=SEED)
    if algo == "nb":
        return GaussianNB(var_smoothing=p["var_smoothing"])
    if algo == "dtree":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **p)
    if algo == "knn":
        return KNeighborsClassifier(n_jobs=n_jobs, **p)
    if algo == "svm":
        return FastSVC(C=p["C"], gamma=p["gamma"])
    if algo == "rf":
        return RandomForestClassifier(class_weight="balanced_subsample",
                                      n_jobs=n_jobs, random_state=SEED, **p)
    if algo == "extra":
        return ExtraTreesClassifier(class_weight="balanced", n_jobs=n_jobs,
                                    random_state=SEED, **p)
    if algo == "hgb":
        return HistGradientBoostingClassifier(random_state=SEED, **p)
    if algo == "xgb":
        if not HAS_XGB:
            return HistGradientBoostingClassifier(random_state=SEED)
        return XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=n_jobs,
                             random_state=SEED, verbosity=0, **p)
    if algo == "lgbm":
        if not HAS_LGB:
            return HistGradientBoostingClassifier(random_state=SEED)
        return LGBMClassifier(class_weight="balanced", n_jobs=n_jobs, verbose=-1,
                              random_state=SEED, **p)
    raise ValueError(algo)


def pipe(est):
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()), ("clf", est)])


# =============================================================== métriques
def ece(y, p, bins=ECE_BINS):
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.digitize(p, edges[1:-1])
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() / len(y)
                     for b in range(bins) if (idx == b).any()))


def boot_ci(y, p, fn=roc_auc_score, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            vals.append(fn(y[i], p[i]))
    return (float(fn(y, p)), float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5))) if vals else (np.nan,) * 3


def paired_delta_ci(y, p_a, p_b, n=N_BOOT, seed=SEED):
    """ΔAUROC (a − b) avec IC bootstrap apparié : la taille d'effet demandée
    par les relecteurs, plus informative qu'une p-value."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            d.append(roc_auc_score(y[i], p_a[i]) - roc_auc_score(y[i], p_b[i]))
    obs = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    return (float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) if d else (obs, np.nan, np.nan)


# =============================================================== données
COMMON5 = ["age", "sex", "sbp", "chol", "diabetes"]


def load_schema(schema, mode="all"):
    """mode :
      all      -> toutes les variables du schéma
      common5  -> UNIQUEMENT les 5 variables observées dans TOUTES les cohortes.
                  C'est l'analyse qui lève le confondant de disponibilité :
                  sans elle, un transfert vers Cleveland mesure aussi l'absence
                  de dbp/smoke/bmi, pas seulement la différence de population.
      nochol   -> sans le cholestérol (sensibilité à la catégorisation NCEP)
    """
    out = {}
    for f in sorted(DIRS[schema].glob("*.csv")):
        df = pd.read_csv(f)
        feats = [c for c in df.columns if c != TARGET]
        if mode == "common5":
            feats = [c for c in feats if c in COMMON5]
        elif mode == "nochol":
            feats = [c for c in feats if c not in ("chol", "chol_mgdl")]
        out[f.stem] = (df[feats], df[TARGET].values.astype(int), feats)
    return out


# =============================================================== étapes
def step_within(data, n_trials, algos, schema, resume):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    path = TABLES / f"v2_{schema}_within.csv"
    done = set()
    if path.exists():
        if resume:
            d = pd.read_csv(path)
            done = set(zip(d.cohort, d.algo))
        else:
            path.unlink()            # repartir proprement sans --resume
    rows, oof = [], {}
    for cname, (X, y, feats) in data.items():
        print(f"\n  [{schema}] {cname}: n={len(y)}, prévalence={y.mean():.3f}")
        for algo in algos:
            if (cname, algo) in done:
                print(f"    {algo:8s} déjà fait"); continue
            cv_in = StratifiedKFold(3, shuffle=True, random_state=SEED)

            def obj(t):
                est = estimator(algo, space(algo, t))
                sc = []
                for tr, va in cv_in.split(X, y):
                    m = pipe(clone(est)).fit(X.iloc[tr], y[tr])
                    sc.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
                return float(np.mean(sc))

            st = optuna.create_study(direction="maximize",
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
            st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
            best = pipe(estimator(algo, space(algo, st.best_trial)))
            cv_out = StratifiedKFold(5, shuffle=True, random_state=SEED)
            p = cross_val_predict(clone(best), X, y, cv=cv_out, method="predict_proba")[:, 1]
            best.fit(X, y)
            joblib.dump(best, MODELS / f"{schema}_{cname}_{algo}.joblib")
            oof[(cname, algo)] = p
            # sauvegarde IMMÉDIATE : avec --resume, une sauvegarde différée en fin
            # de boucle laissait des fichiers manquants, et l'étape transfert
            # retombait silencieusement sur une prédiction EN ÉCHANTILLON.
            np.save(MODELS / f"oof_{schema}_{cname}_{algo}.npy", p)
            a, lo, hi = boot_ci(y, p)
            rows.append({"schema": schema, "cohort": cname, "algo": algo, "n": len(y),
                         "auroc": round(a, 4), "auroc_lo": round(lo, 4), "auroc_hi": round(hi, 4),
                         "auprc": round(average_precision_score(y, p), 4),
                         "brier": round(brier_score_loss(y, p), 4),
                         "ece": round(ece(y, p), 4), "inner_cv_auroc": round(st.best_value, 4)})
            print(f"    {algo:8s} AUROC={a:.3f} [{lo:.3f}-{hi:.3f}]  ECE={rows[-1]['ece']:.3f}")
            # écriture immédiate : une coupure ne coûte qu'un modèle
            pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(),
                                      index=False)
            rows = []
    for k, v in oof.items():
        np.save(MODELS / f"oof_{schema}_{k[0]}_{k[1]}.npy", v)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def step_effect_sizes(data, schema, algos):
    """ΔAUROC de chaque algorithme contre la régression logistique, avec IC."""
    rows = []
    for cname, (X, y, _) in data.items():
        ref = MODELS / f"oof_{schema}_{cname}_logreg.npy"
        if not ref.exists():
            continue
        p_ref = np.load(ref)
        for algo in algos:
            f = MODELS / f"oof_{schema}_{cname}_{algo}.npy"
            if not f.exists() or algo == "logreg":
                continue
            d, lo, hi = paired_delta_ci(y, np.load(f), p_ref)
            rows.append({"schema": schema, "cohort": cname, "algo": algo,
                         "delta_auroc_vs_logreg": round(d, 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "significatif": bool(lo > 0 or hi < 0)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"v2_{schema}_effect_sizes.csv", index=False)
    return df


def step_transfer(data, schema, algos):
    """Matrices de transfert PAR ALGORITHME + moyenne + IC bootstrap."""
    rows = []
    for algo in algos:
        for src, (Xs, ys, fs) in data.items():
            mp = MODELS / f"{schema}_{src}_{algo}.joblib"
            if not mp.exists():
                continue
            model = joblib.load(mp)
            for tgt, (Xt, yt, ft) in data.items():
                common = [c for c in fs if c in ft]
                try:
                    if src == tgt:
                        f = MODELS / f"oof_{schema}_{tgt}_{algo}.npy"
                        if f.exists():
                            p = np.load(f)
                        else:   # jamais de prédiction en échantillon sur la diagonale
                            print(f"    [info] OOF manquant pour {tgt}/{algo} : recalcul")
                            cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
                            p = cross_val_predict(clone(model), Xt, yt, cv=cv,
                                                  method="predict_proba")[:, 1]
                            np.save(f, p)
                    else:
                        Xa = Xt.reindex(columns=fs)
                        p = model.predict_proba(Xa)[:, 1]
                    a, lo, hi = boot_ci(yt, p, n=200)
                except Exception as e:
                    print(f"    [warn] {algo} {src}->{tgt}: {e}"); continue
                rows.append({"schema": schema, "algo": algo, "train": src, "test": tgt,
                             "auroc": round(a, 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"v2_{schema}_transfer_by_model.csv", index=False)
    if df.empty:
        return df, pd.DataFrame()
    mat = df.pivot_table(index="train", columns="test", values="auroc", aggfunc="mean")
    mat.round(4).to_csv(TABLES / f"v2_{schema}_transfer_mean.csv")
    # robustesse au transfert par algorithme : le meilleur "à domicile" voyage-t-il ?
    pen = []
    for algo, g in df.groupby("algo"):
        m = g.pivot_table(index="train", columns="test", values="auroc")
        v = m.values.astype(float)
        diag, off = np.diag(v), v.copy()
        np.fill_diagonal(off, np.nan)
        pen.append({"algo": algo, "within_mean": round(float(np.nanmean(diag)), 4),
                    "transfer_mean": round(float(np.nanmean(off)), 4),
                    "penalty": round(float(np.nanmean(diag) - np.nanmean(off)), 4)})
    pdf = pd.DataFrame(pen).sort_values("penalty")
    pdf["rank_within"] = pdf.within_mean.rank(ascending=False)
    pdf["rank_transfer"] = pdf.transfer_mean.rank(ascending=False)
    pdf.to_csv(TABLES / f"v2_{schema}_transfer_robustness.csv", index=False)
    print("\n  Robustesse au transfert par algorithme :")
    print(pdf.to_string(index=False))
    if len(pdf) > 2:
        t, p = kendalltau(pdf.rank_within, pdf.rank_transfer)
        print(f"  Accord entre rang à domicile et rang en transfert : tau={t:.3f} (p={p:.3f})")
    return df, mat


def shap_importance(model, X, feats, n=1500, seed=SEED):
    """Importance SHAP d'UN modèle sur UN jeu de données (famille unique)."""
    import shap
    rng = np.random.default_rng(seed)
    pos = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X.iloc[pos]
    Xt = model[:-1].transform(Xs)
    try:
        v = shap.TreeExplainer(model[-1]).shap_values(Xt)
        if isinstance(v, list):
            v = v[1]
        v = np.asarray(v)
        if v.ndim == 3:
            v = v[:, :, -1]
        imp = pd.Series(np.abs(v).mean(0), index=feats)
    except Exception:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, Xs, model.predict(Xs), n_repeats=5,
                                   random_state=seed, scoring="roc_auc")
        imp = pd.Series(r.importances_mean, index=feats)
    return imp / max(imp.sum(), 1e-12)


def step_shap_transfer(data, schema, mat):
    """SHAP SOURCE -> CIBLE : diagnostic réellement sans étiquettes.

    Pour chaque paire (A, B) on prend le modèle entraîné sur A, on calcule ses
    attributions sur les données de A puis sur celles de B, et on mesure leur
    concordance. Aucune étiquette de B n'est utilisée. On calcule aussi, à titre
    de comparaison, la concordance entre le modèle de A et le modèle de B
    (protocole de la version 1, qui mélangeait effet population et effet modèle).
    """
    rows = []
    models, imps_self = {}, {}
    for c, (X, y, feats) in data.items():
        mp = MODELS / f"{schema}_{c}_{SHAP_FAMILY}.joblib"
        if not mp.exists():
            continue
        models[c] = (joblib.load(mp), X, feats)
        models[c][0]  # noqa
        imps_self[c] = shap_importance(models[c][0], X, feats)
    for a, b in itertools.combinations(models, 2):
        (ma, Xa, fa), (mb, Xb, fb) = models[a], models[b]
        common = [f for f in fa if f in fb]
        # 1. sans étiquettes : le modèle de A expliqué sur A puis sur B
        ia_on_a = imps_self[a][common]
        ia_on_b = shap_importance(ma, Xb.reindex(columns=fa), fa)[common]
        tau_free_ab = kendalltau(ia_on_a.rank(ascending=False), ia_on_b.rank(ascending=False))[0]
        ib_on_b = imps_self[b][common]
        ib_on_a = shap_importance(mb, Xa.reindex(columns=fb), fb)[common]
        tau_free_ba = kendalltau(ib_on_b.rank(ascending=False), ib_on_a.rank(ascending=False))[0]
        tau_free = float(np.mean([tau_free_ab, tau_free_ba]))
        # 2. protocole v1 : modèle de A contre modèle de B (mélange les effets)
        tau_models = float(kendalltau(ia_on_a.rank(ascending=False),
                                      ib_on_b.rank(ascending=False))[0])
        loss = np.nan
        if not mat.empty and {a, b} <= set(mat.index):
            loss = float(np.mean([mat.loc[b, b] - mat.loc[a, b],
                                  mat.loc[a, a] - mat.loc[b, a]]))
        rows.append({"schema": schema, "pair": f"{a} <-> {b}", "a": a, "b": b,
                     "tau_label_free": round(tau_free, 3),
                     "tau_model_vs_model": round(tau_models, 3),
                     "transfer_loss": round(loss, 4) if loss == loss else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"v2_{schema}_shap_transfer.csv", index=False)
    return df


def mantel_exact(df, tau_col="tau_label_free"):
    """Test de Mantel exact : les paires partagent des cohortes, donc un p de
    Pearson est invalide. Avec k cohortes il n'existe que k! relabellings."""
    coh = sorted(set(df.a) | set(df.b))
    n = len(coh)
    if n < 3:
        return {"note": "moins de 3 cohortes"}
    idx = {c: i for i, c in enumerate(coh)}
    D1 = np.zeros((n, n)); D2 = np.zeros((n, n))
    for _, r in df.iterrows():
        i, j = idx[r.a], idx[r.b]
        D1[i, j] = D1[j, i] = 1 - r[tau_col]
        D2[i, j] = D2[j, i] = r.transfer_loss
    iu = np.triu_indices(n, 1)

    def stat(p):
        return pearsonr(D1[np.ix_(p, p)][iu], D2[iu])[0]

    perms = list(itertools.permutations(range(n)))
    obs = stat(list(range(n)))
    null = np.array([stat(list(p)) for p in perms])
    return {"mantel_r": round(float(obs), 3), "n_cohorts": n,
            "n_permutations": len(perms),
            "p_exact": round(float(np.mean(null >= obs)), 4),
            "p_floor": round(1 / len(perms), 4)}


def step_svm_curve(data, schema):
    """Courbe d'apprentissage du SVM : justifie le plafond à 8 000 par la preuve."""
    big = max(data, key=lambda c: len(data[c][1]))
    X, y, _ = data[big]
    rows = []
    for n in [1000, 2000, 4000, 8000, 16000, 32000]:
        if n > len(y) * 0.8:
            break
        m = pipe(FastSVC(C=1.0, gamma=0.1, max_train_n=n))
        cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
        p = cross_val_predict(m, X, y, cv=cv, method="predict_proba")[:, 1]
        rows.append({"cohort": big, "train_n": n, "auroc": round(roc_auc_score(y, p), 4)})
        print(f"    SVM n={n:6d} -> AUROC={rows[-1]['auroc']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"v2_{schema}_svm_learning_curve.csv", index=False)
    if len(df) > 1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df.train_n, df.auroc, marker="o")
        ax.axvline(SVM_MAX_N, ls="--", c="grey", label=f"cap used ({SVM_MAX_N})")
        ax.set_xscale("log"); ax.set_xlabel("SVM training subsample size")
        ax.set_ylabel("AUROC"); ax.set_title(f"SVM learning curve — {big}")
        ax.grid(alpha=.3); ax.legend()
        plt.tight_layout(); plt.savefig(FIGURES / f"v2_{schema}_svm_learning_curve.png", dpi=300)
        plt.close(fig)
    return df


def step_info_gain(schema="L1"):
    """Gain informationnel 5 -> 8 variables, sur les cohortes complètes."""
    data = load_schema("L1")
    five = ["age", "sex", "sbp", "chol", "diabetes"]
    rows = []
    for c, (X, y, feats) in data.items():
        if not set(["dbp", "smoke", "bmi"]).issubset(feats) or X[["dbp", "smoke", "bmi"]].notna().mean().min() < 0.5:
            continue
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        m = pipe(RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                        n_jobs=4, random_state=SEED))
        p5 = cross_val_predict(clone(m), X[five], y, cv=cv, method="predict_proba")[:, 1]
        p8 = cross_val_predict(clone(m), X[feats], y, cv=cv, method="predict_proba")[:, 1]
        d, lo, hi = paired_delta_ci(y, p8, p5)
        rows.append({"cohort": c, "auroc_5vars": round(roc_auc_score(y, p5), 4),
                     "auroc_8vars": round(roc_auc_score(y, p8), 4),
                     "information_gain": round(d, 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
        print(f"    {c}: 5 var {rows[-1]['auroc_5vars']:.3f} -> 8 var "
              f"{rows[-1]['auroc_8vars']:.3f}  (gain {d:+.3f})")
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "v2_information_gain.csv", index=False)
    return df


def step_nb_folds(data, schema):
    """Distributions des probabilités de Naive Bayes par repli : étaye
    l'explication de l'écart entre AUROC par repli et AUROC poolée."""
    rows = []
    for c, (X, y, _) in data.items():
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        m = pipe(GaussianNB())
        per_fold, pooled = [], np.zeros(len(y))
        for k, (tr, te) in enumerate(cv.split(X, y)):
            mm = clone(m).fit(X.iloc[tr], y[tr])
            p = mm.predict_proba(X.iloc[te])[:, 1]
            pooled[te] = p
            if len(np.unique(y[te])) > 1:
                per_fold.append(roc_auc_score(y[te], p))
            rows.append({"schema": schema, "cohort": c, "fold": k,
                         "p_mean": round(float(p.mean()), 4),
                         "p_sd": round(float(p.std()), 4),
                         "p_q10": round(float(np.percentile(p, 10)), 4),
                         "p_q90": round(float(np.percentile(p, 90)), 4)})
        rows.append({"schema": schema, "cohort": c, "fold": "SUMMARY",
                     "auroc_mean_per_fold": round(float(np.mean(per_fold)), 4),
                     "auroc_pooled": round(float(roc_auc_score(y, pooled)), 4)})
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / f"v2_{schema}_nb_fold_probabilities.csv", index=False)
    return df


# valeurs critiques q_0.05 du test de Nemenyi (test bilatéral, k algorithmes)
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
               7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def step_friedman(schema):
    """Friedman + distance critique de Nemenyi, conservés pour la comparabilité
    avec la version 1 — mais les tailles d'effet restent l'argument principal,
    la puissance de Nemenyi étant faible avec peu de cohortes."""
    f = TABLES / f"v2_{schema}_within.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f).drop_duplicates(["cohort", "algo"], keep="last")
    piv = d.pivot(index="cohort", columns="algo", values="auroc").dropna()
    k, N = piv.shape[1], piv.shape[0]
    if k < 3 or N < 2:
        return {"note": "pas assez de cohortes ou d'algorithmes"}
    stat, p = friedmanchisquare(*[piv[c].values for c in piv.columns])
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    cd = NEMENYI_Q05.get(k, 3.164) * np.sqrt(k * (k + 1) / (6 * N))
    spread = float(ranks.max() - ranks.min())
    res = {"friedman_stat": round(float(stat), 3), "friedman_p": round(float(p), 4),
           "k_algos": k, "n_cohorts": N, "critical_difference": round(float(cd), 3),
           "observed_rank_spread": round(spread, 3),
           "any_pairwise_difference": bool(spread > cd)}
    ranks.round(2).to_csv(TABLES / f"v2_{schema}_friedman_ranks.csv")
    print(f"  Friedman p = {res['friedman_p']}, CD = {res['critical_difference']}, "
          f"écart de rangs = {res['observed_rank_spread']} -> "
          f"différence par paires démontrable : {res['any_pairwise_difference']}")
    print(ranks.round(2).to_string())
    return res


def fig_tau_loss(shap_df, schema):
    d = shap_df.dropna(subset=["transfer_loss"])
    if len(d) < 3:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for col, mk, lab in [("tau_label_free", "o", "Label-free (source model on target data)"),
                         ("tau_model_vs_model", "s", "Model-vs-model (v1 protocol)")]:
        r = pearsonr(d[col], d.transfer_loss)[0]
        ax.scatter(d[col], d.transfer_loss, marker=mk, s=55, label=f"{lab} (r = {r:.2f})")
        z = np.polyfit(d[col], d.transfer_loss, 1)
        xs = np.linspace(d[col].min(), d[col].max(), 20)
        ax.plot(xs, np.polyval(z, xs), lw=1, alpha=.6)
    ax.set_xlabel("Kendall τ between SHAP rankings")
    ax.set_ylabel("Mean transfer loss (AUROC)")
    ax.set_title(f"Explanation agreement versus transfer loss — {schema}")
    ax.grid(alpha=.25); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIGURES / f"v2_{schema}_tau_vs_loss.png", dpi=300)
    plt.close(fig)


# =============================================================== driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="L1", choices=["L1", "L2", "both"])
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--algos", default="")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--features", default="all",
                    choices=["all", "common5", "nochol"],
                    help="all | common5 (lève le confondant de disponibilité) "
                         "| nochol (sensibilité au cholestérol)")
    a = ap.parse_args()
    n_trials = 3 if a.quick else a.trials
    algos = [x.strip() for x in a.algos.split(",") if x.strip()] or ALGOS
    steps = a.steps
    schemas = ["L1", "L2"] if a.schema == "both" else [a.schema]
    summary = {}

    for schema in schemas:
        if not DIRS[schema].exists() or not list(DIRS[schema].glob("*.csv")):
            print(f"[skip] schéma {schema} : aucun fichier dans {DIRS[schema]}")
            continue
        data = load_schema(schema, a.features)
        suffix = {"all": "", "common5": "_c5", "nochol": "_nochol"}[a.features]
        tag = schema + suffix
        print(f"\n{'='*70}\nSCHÉMA {schema} ({a.features}) — {len(data)} cohortes, "
              f"{len(list(data.values())[0][2])} variables, {n_trials} essais\n{'='*70}")
        schema_files = tag

        if steps in ("all", "within"):
            print("\n--- Étape 1/7 : benchmark intra-cohorte ---")
            step_within(data, n_trials, algos, schema_files, a.resume)
        if steps in ("all", "effects"):
            print("\n--- Étape 2/7 : tailles d'effet contre la régression logistique ---")
            print(step_effect_sizes(data, schema_files, algos).to_string(index=False))
        mat = pd.DataFrame()
        if steps in ("all", "transfer"):
            print("\n--- Étape 3/7 : transfert par algorithme ---")
            _, mat = step_transfer(data, schema_files, algos)
            if not mat.empty:
                print("\n  Matrice moyenne :"); print(mat.round(3).to_string())
                v = mat.values.astype(float); off = v.copy(); np.fill_diagonal(off, np.nan)
                summary[f"{schema_files}_penalty"] = round(float(np.nanmean(np.diag(v)) - np.nanmean(off)), 4)
        if steps in ("all", "shap"):
            print("\n--- Étape 4/7 : SHAP source -> cible (sans étiquettes) ---")
            if mat.empty:
                f = TABLES / f"v2_{schema_files}_transfer_mean.csv"
                mat = pd.read_csv(f, index_col=0) if f.exists() else pd.DataFrame()
            sh = step_shap_transfer(data, schema_files, mat)
            if not sh.empty:
                print(sh.to_string(index=False))
                d = sh.dropna(subset=["transfer_loss"])
                if len(d) >= 3:
                    for col in ("tau_label_free", "tau_model_vs_model"):
                        r, p = pearsonr(d[col], d.transfer_loss)
                        rs, _ = spearmanr(d[col], d.transfer_loss)
                        print(f"  {col:22s}: r={r:+.3f} (p={p:.4f})  rho={rs:+.3f}")
                        summary[f"{schema_files}_{col}_r"] = round(float(r), 3)
                    mt = mantel_exact(d)
                    print(f"  Mantel exact : {mt}")
                    summary[f"{schema_files}_mantel"] = mt
                fig_tau_loss(sh, schema_files)
        if steps in ("all", "stats"):
            print("\n--- Étape 5/8 : Friedman + Nemenyi ---")
            summary[f"{schema_files}_friedman"] = step_friedman(schema_files)
        if steps in ("all", "svm"):
            print("\n--- Étape 5/7 : courbe d'apprentissage du SVM ---")
            step_svm_curve(data, schema_files)
        if steps in ("all", "nb"):
            print("\n--- Étape 6/7 : probabilités de Naive Bayes par repli ---")
            step_nb_folds(data, schema_files)

    if steps in ("all", "info"):
        print("\n--- Étape 7/7 : gain informationnel 5 -> 8 variables ---")
        step_info_gain()

    with open(TABLES / "v2_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nTerminé. Résultats dans {TABLES} et {FIGURES}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
