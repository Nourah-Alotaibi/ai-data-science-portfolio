# German Credit Risk: Evaluation and Preprocessing

Rebuilt a German-credit classification notebook as a reproducible pipeline with preprocessing contained inside cross-validation. Compared logistic regression, random forest and SVM and evaluated the accuracy-versus-minority-class tradeoff. The selected model achieved balanced accuracy of 0.742 on the held-out test set versus 0.710 for the baseline, while openly documenting the reduction in overall accuracy and the limits of this historical dataset.

## Why a lower accuracy can still reveal a useful tradeoff

In a credit benchmark with unequal classes, overall accuracy can hide weak detection of the minority class. This project asks how preprocessing and model selection change that tradeoff.

All encoding and scaling move inside cross-validation, then candidates are selected by balanced accuracy. A held-out test set makes the effect on both overall and class-balanced performance visible.

Balanced accuracy improves while ordinary accuracy falls. Reporting both is the result: the model is a transparent historical benchmark, not a validated lending system.

## Status

Executed successfully; measured results saved.

## Method

Used fold-local numeric imputation/scaling and categorical imputation/one-hot encoding inside a pipeline. Compared logistic regression, random forest and SVM using five-fold training cross-validation. Selected by balanced accuracy to account for the minority bad-credit class, then evaluated a held-out 200-record test set.

## Measured results

| Model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Train-only logistic baseline | 0.7800 | 0.7095 | 0.7210 | 0.8040 |
| Selected model | 0.7250 | 0.7417 | 0.7059 | 0.8094 |


## Limits and interpretation

The selected balanced model increased test balanced accuracy from 0.710 to 0.742, while overall accuracy fell from 78.0% to 72.5%. This is a documented tradeoff, not an overall accuracy gain. Historical sensitive attributes are retained for assignment comparability; fairness and deployment suitability were not established. The older 71% result is not a controlled baseline.

## Data

Download UCI Statlog German Credit: https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data . Documentation: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data . Your original Kaggle notebook is https://www.kaggle.com/code/nourahalotaibi1/german-credit-code .

## Reproduce

Run from this project directory in an isolated Python environment. The tabular projects were executed with Python 3.12; the deep-learning projects used Python 3.11 on CPU.

```shell
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python train.py --data data/german.data --out results
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

![test confusion matrix](results/test_confusion_matrix.png)

## Attribution

Portfolio project by Nourah Alotaibi. This package refactors the collected project into a new reproducible workflow. Dataset providers, upstream libraries and pretrained-model authors retain their respective rights. This repository does not grant a new license to third-party data or models.
