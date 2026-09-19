# Cardio_1_2 — information availability, outcome definition and algorithm choice

Code and result tables for:

> **Information availability, outcome definition and algorithm choice compared on one scale: a five-dataset benchmark of cardiovascular prediction models**
> Mohammed Midoun, Asma Amrani. Submitted to *Health Information Science and Systems*, 2026.

Applied cardiovascular prediction is dominated by comparisons between learning
algorithms. Performance also depends on which variables a model is given, on the
clinical question its outcome encodes, and on whether it is used in the population
where it was fitted. This study places those three sources of variation on a single
scale under one protocol.

No raw data are redistributed. See `DATA_SOURCES.md`.

---

## Layout

```
Cardio_1_2/
├── README.md
├── DATA_SOURCES.md               where to obtain the five datasets
├── LICENSE                       MIT, for the code
├── requirements.txt              pinned environment
├── data/
│   ├── harmonized/               five shared predictors, the thin schema
│   └── harmonized_rich/          extended schema, the three UCI datasets only
├── src/
│   ├── config.py                 paths, seed, shared constants
│   ├── harmonize.py              harmonisation, cleaning rules, overlap audit
│   ├── build_datasets.py         assembles the thin and rich schemas
│   ├── models.py                 the ten learners and the tuning pipeline
│   ├── main.py                   the benchmark: tuning, out-of-fold evaluation, transfer
│   ├── analysis_v2.py            discrimination, calibration, decision curves, thresholds
│   ├── robustness.py             Friedman–Nemenyi, bootstrap intervals
│   ├── reviewer_analyses.py      feature ladder, no-tuning run, nested cross-validation
│   ├── descriptives.py           Table 1
│   ├── interpret.py              SHAP summaries
│   └── figures_v2.py             every figure in the paper
└── results/
    ├── tables/                   every CSV cited in the paper
    └── figures/                  every figure in the paper
```

---

## Where each result comes from

### Result tables

Files carry a schema prefix: `L1` is the thin schema, the five predictors shared by
every dataset; `L2` is the rich schema, available only in the three UCI datasets;
`c5` marks the five-dataset run. `nochol` is the sensitivity analysis that removes
cholesterol, whose harmonisation required the strongest assumption.

| Content | Script | Files |
|---|---|---|
| Descriptives, Table 1 | `descriptives.py` | `v2_table1_descriptives.csv`, `descriptive_stats.csv` |
| Harmonisation and the overlap audit | `harmonize.py` | `harmonization_report.csv` |
| Within-dataset discrimination and calibration | `analysis_v2.py` | `v2_L1_within.csv`, `v2_L2_within.csv`, `v2_L1_c5_within.csv`, `within_cohort_metrics.csv` |
| Gain from the richer schema | `analysis_v2.py` | `v2_information_gain.csv` |
| Cross-dataset transfer | `analysis_v2.py` | `v2_L1_transfer_mean.csv`, `v2_L1_transfer_by_model.csv`, `v2_L1_transfer_robustness.csv`, `v2_L1_c5_transfer_*.csv`, `v2_L2_transfer_*.csv`, `cross_cohort_auroc_mean.csv` |
| Transfer decomposition | `robustness.py` | `rev_A_transfer_decomposition.csv` |
| Threshold-explicit counts | `analysis_v2.py` | `v2_L1_threshold_metrics.csv`, `v2_L1_c5_threshold_metrics.csv`, `v2_L2_threshold_metrics.csv`, `threshold_metrics.csv` |
| Net benefit by fold | `analysis_v2.py` | `v2_L1_nb_fold_probabilities.csv` |
| Ranks and effect sizes across datasets | `robustness.py` | `v2_L1_friedman_ranks.csv`, `v2_L1_effect_sizes.csv`, `friedman_nemenyi_ranks.csv` |
| Sensitivity without cholesterol | `analysis_v2.py` | `v2_L1_nochol_within.csv`, `v2_L1_nochol_transfer_*.csv`, `v2_L1_nochol_shap_transfer.csv` |
| Sensitivity without the largest dataset | `analysis_v2.py` | `rev_B_without_kaggle.csv` |
| Recalibration and calibration detail | `analysis_v2.py` | `rev_C_recalibration_L1.csv`, `rev_D_calibration_L1.csv` |
| Where the richer-schema gain comes from | `reviewer_analyses.py` | `rev2_A_feature_ladder.csv` |
| No-tuning run | `reviewer_analyses.py` | `rev2_B_no_tuning_kaggle_cvd70k.csv`, `rev2_B_no_tuning_va_longbeach.csv` |
| Nested cross-validation | `reviewer_analyses.py` | `rev2_C_nested_cv.csv`, `rev2_Cbis_nested_ci_cleveland.csv` |
| Attribution summaries | `interpret.py` | `v2_L1_shap_transfer.csv`, `shap_importance_*.csv` |

### Figures

All figures are written by `figures_v2.py` into `results/figures/`, with the
prefix `figV2_`, except the transfer heat map and the critical-difference
diagram. The per-dataset panels follow the pattern
`figV2_L1_c5_<analysis>_<dataset>.png`, where `<analysis>` is one of `roc`,
`calibration`, `dca` or `confusion`.

> **Before submission.** Several figures still carry an internal title drawn
> inside the image. Springer asks that titles live in the caption, not in the
> artwork. Regenerate them with the title suppressed.

---

## Reproducing

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/harmonize.py           # data/raw -> data/harmonized and harmonized_rich
python src/build_datasets.py
python src/main.py                # tuning and out-of-fold evaluation; the long step
python src/analysis_v2.py
python src/robustness.py
python src/reviewer_analyses.py
python src/descriptives.py
python src/interpret.py
python src/figures_v2.py
```

A single global seed is set in `config.py`. Every reported quantity comes from
out-of-fold predictions under stratified five-fold cross-validation: each patient
is predicted once, by a model that never saw them. Each learner receives the same
50-trial tuning budget.

`main.py` is the expensive step; the others read its outputs and run in minutes.

---

## Data

Five non-overlapping public datasets, 73,634 patients after cleaning.

| Dataset | n | Prevalence | Outcome | Schema |
|---|---|---|---|---|
| Cleveland | 303 | 0.459 | Angiographic coronary disease | thin and rich |
| Hungarian | 294 | 0.361 | Angiographic coronary disease | thin and rich |
| VA Long Beach | 199 | 0.744 | Angiographic coronary disease | thin and rich |
| Framingham | 4,240 | 0.152 | Ten-year incident CHD | thin |
| Kaggle CVD | 68,598 | 0.495 | Disease status at examination | thin |

The **thin schema** is the five predictors recorded in every dataset: age, sex,
systolic pressure, a cholesterol measurement and a glycaemic indicator. The
**rich schema** adds the variables available only in the UCI datasets, including
chest-pain type, resting electrocardiogram and the exercise-test results. The
comparison between the two is the informational contrast reported in the paper.

**Statlog Heart is excluded.** All 270 of its records match a Cleveland record on
age, sex, systolic pressure and serum cholesterol and carry the same label. The
audit is in `harmonize.py` and its output in `harmonization_report.csv`; no other
pair produced more matches than a permutation null would predict. This exclusion
matters: treating Statlog as an independent dataset would have counted the same
patients twice in a transfer analysis.

**VA Long Beach** contains 200 records; one was removed because its systolic
pressure was recorded as zero.

**Kaggle cleaning removed 1,402 of 70,000 records, 2.0%.** The rules were fixed
before modelling and applied without reference to the outcome: systolic pressure
outside 70–250 mmHg, diastolic outside 40–150 mmHg, systolic not exceeding
diastolic, body-mass index outside 12–60 kg/m², or age outside 18–100 years.
`harmonize.py` prints the count removed by each rule.

---

## What the study found

Across the four clinical datasets, the best competing learner gained at most 0.005
AUROC over penalised logistic regression and no interval excluded zero. The only
interval that did came from the largest dataset and amounted to 0.010. The richer
schema was associated with gains of 0.152, 0.185 and 0.076 where it was available.
Transfer between datasets cost 0.052 AUROC overall, and 0.062 across pairs
differing in outcome definition against 0.013 for a change of measurement schema
alone. Within datasets, calibration separated the learners up to eight times more
than discrimination did.

Three caveats travel with those numbers and are stated in the paper. The best
competing learner is selected on the same data used to compute its interval, so
that comparison carries selection optimism; the nested cross-validation and the
no-tuning run are reported to bound it. The nested run does not cover the largest
dataset. And the pairs that differ in outcome definition also differ in population,
prevalence and setting, so 0.062 is not the isolated effect of the definition.

---

## Scope

The five datasets do not answer the same clinical question. They are used as
heterogeneous empirical testbeds for the behaviour of prediction models under a
common protocol, not as a cohort against which a deployable model is validated.
Evaluation is internal and out of fold within each dataset; the transfer analysis
is explicitly labelled as such and its asymmetry is declared.

---

## Related work by the same authors

Two further manuscripts use the same public datasets and share only the
harmonisation step. No analysis, figure, table or result appears in more than one.

- `Cardio_1_1` — class-imbalance correction, calibration and the analytical prior correction
- `Cardio_4` — an audit of feature-attribution methods

---

## Licence and citation

Code released under the MIT licence. The datasets remain under the licences of
their original repositories; see `DATA_SOURCES.md`.

If you use this code, please cite the paper above.

---

## Contact

Mohammed Midoun — mohamed.midoun@univ-mosta.dz
CSTL Laboratory, Department of Mathematics and Computer Science,
University of Mostaganem, Algeria
