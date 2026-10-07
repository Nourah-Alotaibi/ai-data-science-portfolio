# Fashion-MNIST Image Classification with a Regularized CNN

Developed a reproducible Fashion-MNIST image-classification experiment with a regularized CNN and an MLP baseline. Used separate training, validation and official test partitions, training-only normalization, and validation-selected checkpoints. Added learning curves, class-level error analysis and documented results. The selected model achieved 93.14% held-out accuracy and 0.931 macro F1.

## From recognizing simple shapes to separating similar clothes

A pullover, coat and shirt can look similar in a 28×28 grayscale image. This project asks whether a regularized convolutional model can learn more useful visual features than a simple dense-network baseline.

A fixed validation split selects checkpoints. Batch normalization, dropout, AdamW and a learning-rate schedule are added to the CNN, and training statistics are learned without using the official test set.

Learning curves show training progress; the held-out confusion matrix shows which clothing categories remain difficult. This is a benchmark result, with no claim of equivalent performance on real product photography.

## Status

Executed successfully; measured results saved.

## Method

Compared a multilayer perceptron with a convolutional model using batch normalization, dropout, AdamW and cosine learning-rate decay. Used 48,000 training images and 12,000 validation images; selected model checkpoints using validation accuracy. The official 10,000-image test split was reserved for final evaluation. Normalization uses training pixels only.

## Measured results

| Model | Accuracy | Balanced accuracy | Macro F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| MLP baseline | 0.8652 | 0.8652 | 0.8641 | — |
| Regularized CNN | 0.9314 | 0.9314 | 0.9311 | — |


## Limits and interpretation

This is a small-image benchmark, not a validated classifier for real-world product photographs. MLP and CNN have different predefined training budgets. Tutorial notebooks were reference material; this refactor is a new reproducible experiment and does not claim a novel architecture.

## Data

The script downloads the four official Fashion-MNIST files from Zalando Research and checks their published MD5 checksums. Source: https://github.com/zalandoresearch/fashion-mnist . Raw files and model checkpoints are ignored by Git.

## Reproduce

Run from this project directory in an isolated Python environment. The tabular projects were executed with Python 3.12; the deep-learning projects used Python 3.11 on CPU.

```shell
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install -r requirements.txt
python train.py --data data --out results --epochs 10
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

![learning curves](results/learning_curves.png)

![test confusion matrix](results/test_confusion_matrix.png)

## Attribution

Portfolio project by Nourah Alotaibi. This package refactors the collected project into a new reproducible workflow. Dataset providers, upstream libraries and pretrained-model authors retain their respective rights. This repository does not grant a new license to third-party data or models.
