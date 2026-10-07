"""Paired model comparison; accepts classical predictions for the same fixed test rows."""

import argparse
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.stats import binomtest
from common import save_json, plt


def compare(data, features, classical_predictions, classifier, out):
    d = pd.read_csv(data)
    mask = d.split.eq("test").to_numpy()
    y = d.loc[mask, "Sentiment"].to_numpy()
    X = np.load(features)
    linear = np.load(classical_predictions, allow_pickle=False)
    transformer = joblib.load(classifier).predict(X[mask])
    if len(linear) != len(y):
        raise ValueError("Prediction length differs from shared test set.")
    a = linear == y
    b = transformer == y
    wins = int((b & ~a).sum())
    losses = int((a & ~b).sum())
    discordant = wins + losses
    difference = b.astype(float) - a.astype(float)
    rng = np.random.default_rng(42)
    bootstrap = np.array(
        [difference[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
    )
    report = {
        "test_rows": len(y),
        "classical_accuracy": float(a.mean()),
        "transformer_accuracy": float(b.mean()),
        "accuracy_difference_percentage_points": float(difference.mean() * 100),
        "paired_bootstrap_95pct_difference_pp": (
            np.quantile(bootstrap, [0.025, 0.975]) * 100
        ).tolist(),
        "transformer_only_correct": wins,
        "classical_only_correct": losses,
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "exact_mcnemar_two_sided_p": (
            float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
        ),
        "method": "Exact paired correctness test and 2000 row-wise paired bootstrap replicates. No models selected using this analysis.",
        "limitations": [
            "Post-hoc model comparison on the already reported shared test set.",
            "Row-wise calculations assume independent examples; author/time dependencies are unknown.",
            "Multiple testing across other experiments is not corrected; do not infer a general winner.",
        ],
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "paired_comparison.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].bar(
        ["Word + character\nTF-IDF", "Frozen Arabic\ntransformer"],
        [a.mean() * 100, b.mean() * 100],
        color=["#435fa5", "#7864af"],
    )
    axes[0].set(
        ylim=(0, 100),
        ylabel="Held-out accuracy (%)",
        title="Same texts, same test split",
    )
    for i, v in enumerate([a.mean() * 100, b.mean() * 100]):
        axes[0].text(i, v + 1, f"{v:.2f}%", ha="center")
    ci = report["paired_bootstrap_95pct_difference_pp"]
    point = report["accuracy_difference_percentage_points"]
    axes[1].errorbar(
        point,
        0,
        xerr=[[point - ci[0]], [ci[1] - point]],
        fmt="o",
        capsize=8,
        color="#7864af",
    )
    axes[1].axvline(0, color="#777", linestyle="--")
    axes[1].set(
        yticks=[],
        xlabel="Transformer − TF-IDF (percentage points)",
        title="Paired accuracy difference\n95% bootstrap interval",
    )
    fig.tight_layout()
    fig.savefig(out / "paired_comparison.png")
    plt.close(fig)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--classical-predictions", required=True)
    p.add_argument("--classifier", default="models/classifier.joblib")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    compare(a.data, a.features, a.classical_predictions, a.classifier, a.out)
