"""Wisconsin diagnostic benchmark with training-only model selection."""

import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from common import *


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = load_breast_cancer(as_frame=True)
    X = d.data
    y = 1 - d.target
    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    pipe = make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    grid = [
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.001, 0.01, 0.1]},
        {"svc": [LogisticRegression(max_iter=2000)], "svc__C": [0.01, 0.1, 1, 10]},
    ]
    search = GridSearchCV(
        pipe,
        grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=4,
    )
    search.fit(X.iloc[tr], y.iloc[tr])
    baseline = SVC(kernel="linear", C=1, probability=True, random_state=42)
    baseline.fit(X.iloc[tr], y.iloc[tr])
    best = search.best_estimator_
    results = {}
    for name, m in [
        ("Unscaled linear SVM baseline", baseline),
        ("Selected scaled model", best),
    ]:
        results[name] = evaluate(
            y.iloc[te], m.predict(X.iloc[te]), m.predict_proba(X.iloc[te])[:, 1]
        )
    pred = best.predict(X.iloc[te])
    classification_artifacts(y.iloc[te], pred, out, labels=["Benign", "Malignant"])
    pd.DataFrame(search.cv_results_).astype(str).to_csv(
        out / "cv_results.csv", index=False
    )
    report = {
        "dataset": "scikit-learn Wisconsin Diagnostic Breast Cancer",
        "source": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html",
        "rows": len(X),
        "features": len(X.columns),
        "target": "1 = malignant",
        "split_sizes": {"train": len(tr), "test": len(te)},
        "selection": "5-fold training CV ROC-AUC",
        "best_parameters": str(search.best_params_),
        "best_cv_roc_auc": search.best_score_,
        "test_results": results,
        "accuracy_95pct_wilson": accuracy_interval(y.iloc[te], pred),
        "limitations": "Educational benchmark, 114-case test set; no clinical validation. Historical notebook scores use different settings.",
        "environment": versions(),
    }
    save_json(out / "metrics.json", report)
    print(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.out)
