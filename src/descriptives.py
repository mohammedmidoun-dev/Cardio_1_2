"""
descriptives.py — produit les chiffres réels du Tableau 1.

    python src/descriptives.py

Écrit results/tables/v2_table1_descriptives.csv et l'affiche.
Les moyennes du manuscrit doivent venir d'ici, pas d'une estimation.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
VARS = ["age", "sex", "sbp", "dbp", "chol", "smoke", "diabetes", "bmi"]

rows = []
for schema, folder in [("L1", ROOT / "data" / "harmonized"),
                       ("L2", ROOT / "data" / "harmonized_rich")]:
    for f in sorted(folder.glob("*.csv")):
        d = pd.read_csv(f)
        r = {"schema": schema, "cohort": f.stem, "n": len(d),
             "prevalence": round(float(d["target"].mean()), 4)}
        for v in [c for c in d.columns if c != "target"]:
            s = pd.to_numeric(d[v], errors="coerce")
            if s.notna().sum() == 0:
                r[f"{v}_mean"] = "not recorded"
            else:
                r[f"{v}_mean"] = round(float(s.mean()), 2)
                r[f"{v}_sd"] = round(float(s.std()), 2)
                r[f"{v}_complete"] = round(float(s.notna().mean()), 3)
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(OUT / "v2_table1_descriptives.csv", index=False)
cols = ["schema", "cohort", "n", "prevalence"] + [f"{v}_mean" for v in VARS if f"{v}_mean" in df]
print(df[[c for c in cols if c in df]].to_string(index=False))
print(f"\n-> {OUT / 'v2_table1_descriptives.csv'}")
