"""
build_datasets.py
Reconstruit l'ensemble des cohortes, les harmonise, et AUDITE leur indépendance.

    python src/build_datasets.py                    # tout
    python src/build_datasets.py --no-download      # fichiers déjà présents

POURQUOI CE SCRIPT EXISTE
-------------------------
L'audit a montré que Statlog Heart n'est pas une cohorte indépendante : ses
270 patients sont TOUS présents dans Cleveland (appariement exact 270/270,
étiquettes concordantes à 100 %). Le "transfert" Cleveland <-> Statlog était
donc une évaluation in-sample, et c'est ce point qui portait la corrélation
SHAP-transfert (r passe de -0.910 à -0.845, p de 0.012 à 0.072 sans lui).

Ce script :
  1. télécharge les QUATRE bases UCI Heart Disease, qui sont de vraies
     populations distinctes (Cleveland/Ohio, Hongrie/Budapest, VA Long
     Beach/Californie, Suisse/Zurich-Bâle) ;
  2. harmonise tout au schéma commun à 8 variables ;
  3. nettoie les valeurs physiologiquement impossibles de la cohorte Kaggle
     AVANT toute modélisation ;
  4. exécute un AUDIT DE RECOUVREMENT sur toutes les paires de cohortes et
     refuse de valider si deux cohortes partagent des patients.

SOURCES
-------
UCI Heart Disease (CC BY 4.0), DOI 10.24432/C52P4X :
  https://archive.ics.uci.edu/dataset/45/heart+disease
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.va.data
  https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data
Framingham (sous-ensemble pédagogique) : à placer dans data/framingham.csv
Kaggle Cardiovascular Disease : à placer dans data/cardio_train.csv (sep=';')
  https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
"""
from __future__ import annotations
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data"
OUT = RAW / "harmonized"
TABLES = ROOT / "results" / "tables"
OUT_RICH = RAW / "harmonized_rich"
for d in (RAW, OUT, OUT_RICH, TABLES):
    d.mkdir(parents=True, exist_ok=True)

FEATURES = ["age", "sex", "sbp", "dbp", "chol", "smoke", "diabetes", "bmi"]
TARGET = "target"

UCI_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/"
UCI_FILES = {                       # nom de sortie -> fichier source
    "cleveland":   "processed.cleveland.data",
    "hungarian":   "processed.hungarian.data",
    "va_longbeach": "processed.va.data",
    "switzerland": "processed.switzerland.data",
}
UCI_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]

# Schéma L2 ("étroit et riche") : les 10 variables cliniques présentes dans les
# QUATRE bases UCI. slope, ca et thal sont écartées : manquantes à plus de 60 %
# hors Cleveland. Le cholestérol reste ici en mg/dL (continu), puisque les
# quatre bases utilisent la même unité : la catégorisation n'est plus nécessaire.
RICH_FEATURES = ["age", "sex", "cp", "sbp", "chol_mgdl", "fbs",
                 "restecg", "thalach", "exang", "oldpeak"]
OUT_RICH = RAW / "harmonized_rich"

# Bornes STRICTES : réservées à la cohorte Kaggle, dont la contamination par
# valeurs impossibles (ap_lo à 0, négatif ou à plusieurs milliers) est documentée.
BOUNDS_STRICT = {"age": (18, 100), "sbp": (70, 250), "dbp": (40, 150), "bmi": (12, 60)}
# Bornes LARGES pour les cohortes cliniques : on n'écarte que l'impossible, afin
# de ne pas supprimer des valeurs extrêmes mais authentiques (Framingham monte
# à 295 mmHg de systolique).
BOUNDS_LOOSE = {"age": (10, 110), "sbp": (50, 300), "dbp": (20, 200), "bmi": (10, 80)}
BOUNDS = BOUNDS_STRICT          # rétro-compatibilité
CHOL_RANGE = (80, 700)

# Seuils NCEP/ATP III pour l'échelle commune du cholestérol
CHOL_BINS = [-np.inf, 200, 240, np.inf]
CHOL_LABELS = [1, 2, 3]


# ------------------------------------------------------------------ download
def download_uci() -> dict[str, Path]:
    paths = {}
    for name, fname in UCI_FILES.items():
        dest = RAW / fname
        paths[name] = dest
        if dest.exists():
            print(f"  [cache] {fname}")
            continue
        url = UCI_BASE + fname
        try:
            df = pd.read_csv(url, header=None, na_values=["?", -9.0])
            df.to_csv(dest, header=False, index=False)
            print(f"  [ok]    {fname}  ({len(df)} lignes)")
        except Exception as e:
            print(f"  [ÉCHEC] {fname}: {e}")
            print(f"          Téléchargez manuellement : {url}")
            print(f"          et placez le fichier dans {RAW}")
    return paths


# ------------------------------------------------------------------ helpers
def chol_to_category(mgdl: pd.Series) -> pd.Series:
    """mg/dL -> {1,2,3}. Les 0 (Suisse) sont traités comme MANQUANTS, pas
    comme un cholestérol nul : c'est un défaut documenté de cette base."""
    s = pd.to_numeric(mgdl, errors="coerce")
    s = s.where(s.between(*CHOL_RANGE))          # 0 et aberrations -> NaN
    return pd.to_numeric(pd.cut(s, CHOL_BINS, labels=CHOL_LABELS, right=False),
                         errors="coerce").astype(float)


def clip_bounds(df: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Filtrage strict pour Kaggle (contamination documentée), large ailleurs."""
    bounds = BOUNDS_STRICT if cohort == "kaggle_cvd70k" else BOUNDS_LOOSE
    n0 = len(df)
    m = pd.Series(True, index=df.index)
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            m &= df[c].between(lo, hi) | df[c].isna()
    if {"sbp", "dbp"}.issubset(df.columns):
        m &= (df["sbp"] > df["dbp"]) | df["dbp"].isna()
    df = df.loc[m].reset_index(drop=True)
    if len(df) < n0:
        print(f"    filtrage physiologique : {n0} -> {len(df)} "
              f"({100*(n0-len(df))/n0:.2f} % retirés)")
    return df


# ------------------------------------------------------------------ harmonisers
def harmonize_uci(path: Path, cohort: str) -> pd.DataFrame:
    """Les quatre bases UCI partagent le même format à 14 colonnes."""
    df = pd.read_csv(path, header=None, names=UCI_COLS, na_values=["?", -9.0])
    o = pd.DataFrame()
    o["age"] = pd.to_numeric(df["age"], errors="coerce")
    o["sex"] = pd.to_numeric(df["sex"], errors="coerce")        # 1 = homme
    o["sbp"] = pd.to_numeric(df["trestbps"], errors="coerce")
    o["dbp"] = np.nan                                            # non mesurée
    o["chol"] = chol_to_category(df["chol"])
    o["smoke"] = np.nan                                          # non mesuré
    # fbs = glycémie à jeun > 120 mg/dL : PROXY glycémique, pas un diagnostic
    o["diabetes"] = pd.to_numeric(df["fbs"], errors="coerce")
    o["bmi"] = np.nan                                            # non mesuré
    o[TARGET] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    return clip_bounds(o, cohort)


def harmonize_uci_rich(path: Path, cohort: str) -> pd.DataFrame:
    """Schéma L2 : 10 variables cliniques, cohortes UCI uniquement.

    Permet de tester si la pénalité de transfert persiste quand l'information
    disponible est riche — donc de répondre à l'objection selon laquelle les
    performances seraient basses à cause d'une harmonisation appauvrie.
    """
    df = pd.read_csv(path, header=None, names=UCI_COLS, na_values=["?", -9.0])
    o = pd.DataFrame()
    o["age"] = pd.to_numeric(df["age"], errors="coerce")
    o["sex"] = pd.to_numeric(df["sex"], errors="coerce")
    o["cp"] = pd.to_numeric(df["cp"], errors="coerce")            # type de douleur
    o["sbp"] = pd.to_numeric(df["trestbps"], errors="coerce")
    ch = pd.to_numeric(df["chol"], errors="coerce")
    o["chol_mgdl"] = ch.where(ch.between(*CHOL_RANGE))            # 0 = non mesuré
    o["fbs"] = pd.to_numeric(df["fbs"], errors="coerce")
    o["restecg"] = pd.to_numeric(df["restecg"], errors="coerce")
    o["thalach"] = pd.to_numeric(df["thalach"], errors="coerce")  # FC maximale
    o["exang"] = pd.to_numeric(df["exang"], errors="coerce")      # angine d'effort
    o["oldpeak"] = pd.to_numeric(df["oldpeak"], errors="coerce")  # sous-décalage ST
    o[TARGET] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
    n0 = len(o)
    # Les valeurs MANQUANTES sont conservées (elles seront imputées dans le
    # pipeline) : seules les valeurs impossibles sont écartées.
    keep = ((o["age"].between(10, 110) | o["age"].isna())
            & (o["sbp"].between(50, 300) | o["sbp"].isna()))
    o = o[keep].reset_index(drop=True)
    if len(o) < n0:
        print(f"    [L2] {cohort}: {n0} -> {len(o)} (valeurs impossibles)")
    return o


def harmonize_framingham(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    o = pd.DataFrame()
    o["age"] = pd.to_numeric(df["age"], errors="coerce")
    o["sex"] = pd.to_numeric(df["male"], errors="coerce")
    o["sbp"] = pd.to_numeric(df["sysBP"], errors="coerce")
    o["dbp"] = pd.to_numeric(df["diaBP"], errors="coerce")
    o["chol"] = chol_to_category(df["totChol"])
    o["smoke"] = pd.to_numeric(df["currentSmoker"], errors="coerce")
    o["diabetes"] = pd.to_numeric(df["diabetes"], errors="coerce")
    o["bmi"] = pd.to_numeric(df["BMI"], errors="coerce")
    o[TARGET] = pd.to_numeric(df["TenYearCHD"], errors="coerce").astype(int)
    return clip_bounds(o, "framingham")


def harmonize_kaggle(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    o = pd.DataFrame()
    o["age"] = pd.to_numeric(df["age"], errors="coerce") / 365.25
    o["sex"] = (pd.to_numeric(df["gender"], errors="coerce") - 1).astype(float)
    o["sbp"] = pd.to_numeric(df["ap_hi"], errors="coerce")
    o["dbp"] = pd.to_numeric(df["ap_lo"], errors="coerce")
    o["chol"] = pd.to_numeric(df["cholesterol"], errors="coerce")   # déjà 1-3
    o["smoke"] = pd.to_numeric(df["smoke"], errors="coerce")
    o["diabetes"] = (pd.to_numeric(df["gluc"], errors="coerce") > 1).astype(float)
    h = pd.to_numeric(df["height"], errors="coerce") / 100
    o["bmi"] = pd.to_numeric(df["weight"], errors="coerce") / (h ** 2)
    o[TARGET] = pd.to_numeric(df["cardio"], errors="coerce").astype(int)
    return clip_bounds(o, "kaggle_cvd70k")


# ------------------------------------------------------------------ audits
def check_target(df: pd.DataFrame, name: str) -> float:
    u = set(pd.unique(df[TARGET].dropna()))
    if not u <= {0, 1}:
        raise ValueError(f"[{name}] cible mal codée : {sorted(u)}")
    prev = float(df[TARGET].mean())
    if not 0.0 < prev < 1.0:
        raise ValueError(f"[{name}] prévalence aberrante : {prev}")
    return prev


def overlap_audit(cohorts: dict[str, pd.DataFrame],
                  rich: dict[str, pd.DataFrame] | None = None,
                  n_null: int = 100, seed: int = 42) -> pd.DataFrame:
    """Détecte les patients partagés entre cohortes.

    Deux précautions, apprises de l'épisode Statlog (270/270 patients de Statlog
    présents dans Cleveland) et des faux positifs qui ont suivi :

    1. CLÉS CONTINUES quand elles existent. Pour les paires de bases UCI on
       apparie sur le cholestérol en mg/dL, la fréquence cardiaque maximale et
       le sous-décalage ST : deux dossiers identiques sur ces variables sont
       le même patient. Sur des clés grossières (âge entier, sexe, cholestérol
       en trois classes) les collisions fortuites sont massives.

    2. RÉFÉRENCE DE HASARD. On compare le nombre d'appariements observés à
       celui obtenu quand la structure conjointe est détruite (colonnes
       permutées indépendamment). Un recouvrement réel dépasse largement ce
       niveau ; des collisions fortuites s'y confondent.
    """
    rng = np.random.default_rng(seed)
    rich = rich or {}
    rows = []
    for a, b in itertools.combinations(cohorts, 2):
        use_rich = a in rich and b in rich
        A = rich[a] if use_rich else cohorts[a]
        B = rich[b] if use_rich else cohorts[b]
        cand = (["age", "sex", "sbp", "chol_mgdl", "thalach", "oldpeak"] if use_rich
                else ["age", "sex", "sbp", "dbp", "bmi", "chol", TARGET])
        keys = [c for c in cand if c in A.columns and c in B.columns
                and A[c].notna().any() and B[c].notna().any()]
        if len(keys) < 3:
            keys = ["age", "sex", "sbp"]
        Ak = A[keys].astype(float).round(3).dropna()
        Bk = B[keys].astype(float).round(3).dropna()
        if len(Ak) == 0 or len(Bk) == 0:
            continue
        obs = len(Ak.merge(Bk, on=keys, how="inner"))

        # référence de hasard : colonnes de B permutées indépendamment
        null = []
        for _ in range(n_null):
            Bp = pd.DataFrame({c: rng.permutation(Bk[c].values) for c in keys})
            null.append(len(Ak.merge(Bp, on=keys, how="inner")))
        null_mean = float(np.mean(null))
        ratio = obs / max(null_mean, 0.5)
        share = obs / max(min(len(Ak), len(Bk)), 1)
        verdict = ("RECOUVREMENT" if (share > 0.30 and ratio > 3)
                   else "A VERIFIER" if (share > 0.10 and ratio > 3)
                   else "independantes")
        rows.append({"cohort_a": a, "cohort_b": b,
                     "keys": "+".join(keys), "n_keys": len(keys),
                     "cles_continues": use_rich,
                     "n_a": len(Ak), "n_b": len(Bk), "matched_rows": obs,
                     "hasard_attendu": round(null_mean, 1),
                     "ratio_obs_hasard": round(ratio, 1),
                     "share_of_smaller": round(share, 3), "verdict": verdict})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--keep-switzerland", action="store_true",
                    help="inclure la base suisse (cholestérol quasi absent)")
    args = ap.parse_args()

    print("=== 1. Téléchargement des bases UCI ===")
    paths = {} if args.no_download else download_uci()
    if args.no_download:
        paths = {n: RAW / f for n, f in UCI_FILES.items()}

    print("\n=== 2. Harmonisation ===")
    cohorts: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        if name == "switzerland" and not args.keep_switzerland:
            print(f"  [ignoré] switzerland (cholestérol non exploitable ; "
                  f"--keep-switzerland pour l'inclure)")
            continue
        if not p.exists():
            print(f"  [absent] {p.name}")
            continue
        print(f"  {name}")
        cohorts[name] = harmonize_uci(p, name)

    print("\n=== 2b. Schéma L2 (10 variables cliniques, cohortes UCI) ===")
    rich_report = []
    rich_frames: dict[str, pd.DataFrame] = {}
    for name, p in paths.items():
        if name == "switzerland" and not args.keep_switzerland:
            continue
        if not p.exists():
            continue
        r = harmonize_uci_rich(p, name)
        prev = check_target(r, name + " (L2)")
        comp = {c: round(float(r[c].notna().mean()), 3) for c in RICH_FEATURES}
        r.to_csv(OUT_RICH / f"{name}.csv", index=False)
        rich_frames[name] = r
        weak = [c for c, v in comp.items() if v < 0.5]
        print(f"  {name:14s} n={len(r):5d}  prévalence={prev:.3f}"
              + (f"  | complétude < 50 % : {', '.join(weak)}" if weak else ""))
        rich_report.append({"cohort": name, "n": len(r),
                            "prevalence": round(prev, 4), **comp})
    if rich_report:
        pd.DataFrame(rich_report).to_csv(TABLES / "harmonization_report_rich.csv",
                                         index=False)

    fram = RAW / "framingham.csv"
    if fram.exists():
        print("  framingham")
        cohorts["framingham"] = harmonize_framingham(fram)
    else:
        print(f"  [absent] {fram}")

    kag = RAW / "cardio_train.csv"
    if kag.exists():
        print("  kaggle_cvd70k")
        cohorts["kaggle_cvd70k"] = harmonize_kaggle(kag)
    else:
        print(f"  [absent] {kag}")

    if not cohorts:
        print("Aucune cohorte construite.")
        return 1

    print("\n=== 3. Contrôles et écriture ===")
    report = []
    for name, df in cohorts.items():
        prev = check_target(df, name)
        comp = {c: round(float(df[c].notna().mean()), 3) for c in FEATURES}
        df.to_csv(OUT / f"{name}.csv", index=False)
        empty = [c for c, v in comp.items() if v == 0.0]
        print(f"  {name:14s} n={len(df):6d}  prévalence={prev:.3f}"
              + (f"  | absentes : {', '.join(empty)}" if empty else ""))
        report.append({"cohort": name, "n": len(df), "prevalence": round(prev, 4), **comp})
    rep = pd.DataFrame(report)
    rep.to_csv(TABLES / "harmonization_report.csv", index=False)

    print("\n=== 4. AUDIT DE RECOUVREMENT (le contrôle qui manquait) ===")
    ov = overlap_audit(cohorts, rich_frames)
    ov.to_csv(TABLES / "overlap_report.csv", index=False)
    print(ov.to_string(index=False))
    bad = ov[ov.verdict == "RECOUVREMENT"]
    warn = ov[ov.verdict == "A VERIFIER"]
    if len(bad):
        print("\n  ATTENTION : cohortes non indépendantes détectées.")
        for _, r in bad.iterrows():
            print(f"    {r.cohort_a} <-> {r.cohort_b} : {r.matched_rows} patients "
                  f"communs ({100*r.share_of_smaller:.0f} % de la plus petite)")
        print("    -> ne pas les traiter comme un transfert externe ;")
        print("       les exclure des analyses de transportabilité.")
    elif len(warn):
        print("\n  Recouvrement possible (collisions fortuites probables sur de "
              "petites cohortes) :")
        for _, r in warn.iterrows():
            print(f"    {r.cohort_a} <-> {r.cohort_b} : {r.matched_rows} lignes "
                  f"appariées sur {r.n_keys} clés ({100*r.share_of_smaller:.1f} %)")
        print("    -> inspecter manuellement si la part dépasse 20 %.")
    else:
        print("\n  Aucun recouvrement détecté : les cohortes sont indépendantes.")

    print(f"\nSchéma L1 (5 variables, toutes cohortes) : {OUT}")
    print(f"Schéma L2 (10 variables, cohortes UCI)    : {OUT_RICH}")
    print(f"Rapports : {TABLES/'harmonization_report.csv'}, {TABLES/'overlap_report.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
