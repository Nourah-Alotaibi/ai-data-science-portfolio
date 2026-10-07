"""Chronological weather clustering with circular wind-direction features."""

import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score
from common import *

SOURCE = "https://media.githubusercontent.com/media/devmukul44/dse200x-python-for-datascience/75a4669cb5c97c5487d3c7e8eb3f4481f4e98ee6/Week7-MachineLearning/weather/minute_weather.csv"


def run(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw = pd.read_csv(data)
    n = len(raw)
    if "hpwren_timestamp" in raw:
        raw["hpwren_timestamp"] = pd.to_datetime(raw.hpwren_timestamp, errors="coerce")
        raw = raw.dropna(subset=["hpwren_timestamp"]).sort_values("hpwren_timestamp")
    else:
        raise ValueError("Timestamp column is required for chronological evaluation.")
    # Sample every tenth minute, matching the scope of the original coursework.
    d = raw.iloc[::10].copy().reset_index(drop=True)
    cols = [
        "air_pressure",
        "air_temp",
        "avg_wind_direction",
        "avg_wind_speed",
        "max_wind_direction",
        "max_wind_speed",
        "min_wind_direction",
        "min_wind_speed",
        "relative_humidity",
    ]
    missing = [c for c in cols if c not in d]
    if missing:
        raise ValueError("Missing columns: " + str(missing))
    x = d[cols].apply(pd.to_numeric, errors="coerce")
    original = x.copy()
    for c in [c for c in cols if "direction" in c]:
        radians = np.deg2rad(x.pop(c) % 360)
        x[c + "_sin"] = np.sin(radians)
        x[c + "_cos"] = np.cos(radians)
    n1 = int(len(d) * 0.6)
    n2 = int(len(d) * 0.8)
    pre = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    tr = pre.fit_transform(x.iloc[:n1])
    va = pre.transform(x.iloc[n1:n2])
    te = pre.transform(x.iloc[n2:])
    rows = []
    for k in range(2, 13):
        m = MiniBatchKMeans(n_clusters=k, random_state=SEED, n_init=10, batch_size=2048)
        m.fit(tr)
        labels = m.predict(va)
        score = (
            silhouette_score(
                va, labels, sample_size=min(2000, len(va)), random_state=SEED
            )
            if len(np.unique(labels)) > 1
            else -1
        )
        rows.append({"k": k, "validation_silhouette": float(score)})
        print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    k = int(table.sort_values("validation_silhouette", ascending=False).iloc[0]["k"])
    # Refit preprocessing and chosen K on development data only, then evaluate later period.
    dev = pre.fit_transform(x.iloc[:n2])
    te = pre.transform(x.iloc[n2:])
    m = MiniBatchKMeans(
        n_clusters=k, random_state=SEED, n_init=10, batch_size=2048
    ).fit(dev)
    labels = m.predict(te)
    stability = []
    for seed in [7, 123]:
        other = MiniBatchKMeans(
            n_clusters=k, random_state=seed, n_init=10, batch_size=2048
        ).fit(dev)
        stability.append(float(adjusted_rand_score(labels, other.predict(te))))
    profile = original.iloc[n2:].assign(cluster=labels)
    # Circular directions need circular means, rather than arithmetic means of degrees.
    profile = (
        profile.drop(columns=[c for c in cols if "direction" in c])
        .groupby("cluster")
        .mean()
    )
    profile["observations"] = pd.Series(labels).value_counts()
    profile.to_csv(out / "test_cluster_profiles.csv")
    table.to_csv(out / "validation_k.csv", index=False)
    fig, ax = plt.subplots()
    ax.plot(table.k, table.validation_silhouette, marker="o", color="#435fa5")
    ax.set(
        xlabel="Number of clusters",
        ylabel="Validation silhouette",
        title="Select K using a later validation period",
    )
    ax.axvline(k, color="#b06543", linestyle="--", label=f"Selected K={k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "select_k.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    sample = (
        original.iloc[n2:]
        .assign(cluster=labels)
        .sample(min(4000, len(te)), random_state=SEED)
    )
    colors=plt.get_cmap('tab10')
    for cluster, observations in sample.groupby('cluster'):
        ax.scatter(observations.air_temp, observations.relative_humidity,
                   color=colors(int(cluster)%10), s=8, alpha=.5,
                   label=f'Cluster {cluster}')
    ax.set(
        xlabel="Air temperature (source units)",
        ylabel="Relative humidity (%)",
        title="Held-out weather clusters",
    )
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "weather_clusters.png")
    plt.close(fig)
    models = out.parent / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(
        {"preprocess": pre, "clusterer": m, "feature_columns": list(x)},
        models / "model.joblib",
    )
    report = {
        "source_url": SOURCE,
        "source_note": "Public mirror of the course dataset; not retrieved from your private Colab. Mirror provenance and exact equivalence to your original file are unverified.",
        "input_sha256": sha256(data),
        "source_rows": n,
        "sampled_rows": len(d),
        "split_sizes": {"train": n1, "validation": n2 - n1, "test": len(d) - n2},
        "selected_k": k,
        "selection_metric": "validation silhouette",
        "test_silhouette": float(
            silhouette_score(
                te, labels, sample_size=min(2000, len(te)), random_state=SEED
            )
        ),
        "test_davies_bouldin": float(davies_bouldin_score(te, labels)),
        "stability_adjusted_rand_other_seeds": stability,
        "runtime_seconds": time.time() - start,
        "environment": versions(),
        "limitations": [
            "Unsupervised clustering has no labeled accuracy.",
            "Chronological split measures later-period behavior at one station; not geographic generalization.",
            "Rainfall features omitted to match the original weather-state feature set.",
            "Clusters describe mathematical groupings, not verified meteorological regimes.",
        ],
    }
    save_json(out / "metrics.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results")
    a = p.parse_args()
    run(a.data, a.out)
