"""Credit-risk benchmark with fold-local mixed-type preprocessing."""

import argparse, urllib.request, time
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from common import *

COLS = [
    "CheckingStatus",
    "Duration",
    "CreditHistory",
    "Purpose",
    "CreditAmount",
    "Savings",
    "EmploymentSince",
    "InstallmentRate",
    "PersonalStatusSex",
    "OtherDebtors",
    "ResidenceSince",
    "Property",
    "Age",
    "OtherInstallPlans",
    "Housing",
    "ExistingCredits",
    "Job",
    "NumDependents",
    "Telephone",
    "ForeignWorker",
    "Creditability",
]


def run(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data, sep=r"\s+", header=None, names=COLS)
    y = (df.pop("Creditability") == 2).astype(int)
    X = df
    tr, te = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
    )
    categorical = X.select_dtypes("object").columns.tolist()
    numeric = X.select_dtypes("number").columns.tolist()
    prep = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric,
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
                categorical,
            ),
        ]
    )
    pipe = make_pipeline(prep, LogisticRegression(max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    grid = [
        {
            "logisticregression": [LogisticRegression(max_iter=2000)],
            "logisticregression__C": [0.01, 0.1, 1, 10],
            "logisticregression__class_weight": [None, "balanced"],
        },
        {
            "logisticregression": [
                RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=42)
            ],
            "logisticregression__max_depth": [5, None],
            "logisticregression__min_samples_leaf": [3, 8],
            "logisticregression__class_weight": ["balanced"],
        },
        {
            "logisticregression": [SVC(probability=True, random_state=42)],
            "logisticregression__C": [0.1, 1, 10],
            "logisticregression__gamma": ["scale", 0.01],
            "logisticregression__class_weight": ["balanced"],
        },
    ]
    search = GridSearchCV(pipe, grid, scoring="balanced_accuracy", cv=cv, n_jobs=1)
    search.fit(X.iloc[tr], y.iloc[tr])
    baseline = make_pipeline(prep, LogisticRegression(max_iter=2000))
    baseline.fit(X.iloc[tr], y.iloc[tr])
    best = search.best_estimator_
    results = {}
    for name, m in [
        ("Train-only logistic baseline", baseline),
        ("Selected model", best),
    ]:
        results[name] = evaluate(
            y.iloc[te], m.predict(X.iloc[te]), m.predict_proba(X.iloc[te])[:, 1]
        )
    pred = best.predict(X.iloc[te])
    classification_artifacts(
        y.iloc[te], pred, out, labels=["Good credit", "Bad credit"]
    )
    pd.DataFrame(search.cv_results_).drop(columns=["params"]).astype(str).to_csv(
        out / "cv_results.csv", index=False
    )
    report = {
        "dataset": "UCI Statlog German Credit",
        "source": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        "input_sha256": sha256(data),
        "rows": len(X),
        "features": len(X.columns),
        "target": "1 = bad credit",
        "split_sizes": {"train": len(tr), "test": len(te)},
        "selection": "5-fold training CV balanced accuracy",
        "best_parameters": str(search.best_params_),
        "best_cv_balanced_accuracy": search.best_score_,
        "test_results": results,
        "accuracy_95pct_wilson": accuracy_interval(y.iloc[te], pred),
        "limitations": "Single split and small historical benchmark. Sensitive attributes retained for assignment comparability; not validated for lending decisions. Original 71% used leaky preprocessing and is not the improvement baseline.",
        "environment": versions(),
    }
    save_json(out / "metrics.json", report)
    print(report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.data, a.out)
