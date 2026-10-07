# Weather Pattern Clustering with Temporal Validation

Refactored a weather-clustering workflow for 1.59 million minute-level records, using a 158,726-row sample. Added circular wind-direction features, temporal validation, training-only preprocessing and data-driven selection of cluster count. The chosen two-cluster solution achieved a held-out silhouette score of 0.325, with stability checks across random seeds. Documented the dataset mirror and limitations of interpreting unsupervised groups.

## Discovering weather patterns without assuming twelve clusters

Minute-level weather records contain cycles, missing measurements and directional values. The question is whether a small set of repeatable patterns appears in a later time period, rather than merely finding groups in the same records used for fitting.

Wind directions become circular sine/cosine features. The pipeline separates time periods, fits preprocessing on earlier data and compares 2–12 clusters on validation silhouette before checking the final period.

Two clusters separate best under this feature set and metric. Stability across random seeds is high, but moderate silhouette and single-station data keep the interpretation exploratory.

## Status

Executed successfully; measured results saved.

## Method

Sampled every tenth row from 1,587,257 minute-level observations. Encoded wind directions as sine/cosine, excluded identifier columns, and fitted imputation and scaling on an earlier time period. Selected K from 2–12 using validation silhouette, then assessed a later test period and stability across random seeds.

## Measured results

See `results/metrics.json` and the accompanying aggregate CSV files for all measured findings.

## Limits and interpretation

Clustering has no labeled classification accuracy. The selected two-cluster solution has test silhouette 0.325 and cross-seed adjusted Rand agreement of 0.977 and 0.965. These summarize mathematical separation/stability, not meteorological ground truth or performance at other stations. Original notebook is collaborative coursework; preserve original collaborator credits if publishing the coursework.

## Data

Use minute_weather.csv with hpwren_timestamp and the weather columns specified in analyze.py. This execution used a public mirror of the course dataset, not your private Colab. Download URL is recorded in results/metrics.json. The Git LFS object SHA-256 is b0e075e9655b505afe40b75b53dcc7eeab70677ead4c3e98c58cbd8b8be82177. Mirror provenance and equivalence to your original private file remain unverified.

## Reproduce

Run from this project directory in an isolated Python environment. The tabular projects were executed with Python 3.12; the deep-learning projects used Python 3.11 on CPU.

```shell
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python analyze.py --data data/minute_weather.csv --out results
```

`analysis.ipynb` is an executed results-review notebook. It displays saved outputs by default; its optional training cell can rerun the experiment after the dataset path is configured. Training scripts were executed separately to produce the recorded results. Raw datasets, downloaded encoders and trained model files are excluded from Git. Only load model files you created or trust.

## Files

- `train.py` or `analyze.py`: complete experiment or analysis.
- `common.py`: metric, figure and reproducibility helpers.
- `results/metrics.json`: measured outcomes, data fingerprint and environment.
- `results/*.csv` and `results/*.png`: aggregate tables and figures.
- `analysis.ipynb`: reproducible review and optional rerun instructions.
- `project-description.md`: LinkedIn-ready description and skills.

## Figures

![select k](results/select_k.png)

![weather clusters](results/weather_clusters.png)

## Attribution

Portfolio project by Nourah Alotaibi. This package refactors the collected project into a new reproducible workflow. Dataset providers, upstream libraries and pretrained-model authors retain their respective rights. This repository does not grant a new license to third-party data or models.
