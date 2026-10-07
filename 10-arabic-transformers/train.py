"""Frozen Arabic BERT features with a supervised sentiment classification head."""

import argparse, time, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch, joblib
from transformers import AutoTokenizer, AutoModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from common import *

MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-da"


def run(data, out, model_name=MODEL, batch_size=32):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(8)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d = pd.read_csv(data)
    required = {"Text", "Sentiment", "normalized", "split"}
    if not required.issubset(d):
        raise ValueError(
            "Use prepared.csv from project 03; required columns: " + str(required)
        )
    if d.normalized.duplicated().any():
        raise ValueError("Duplicate normalized texts would invalidate shared split.")
    if set(d.split) != {"train", "validation", "test"}:
        raise ValueError("Expected fixed train/validation/test partition.")
    cache = out.parent / "data"
    cache.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(
        (sha256(data) + model_name + "max96-mean-pooling-int8-v1").encode()
    ).hexdigest()
    features = cache / (fingerprint + ".npy")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(model_name, trust_remote_code=False)
    encoder.eval()
    revision = getattr(encoder.config, "_commit_hash", None)
    # CPU dynamic int8 linear layers reduce inference cost; encoder remains frozen.
    encoder = torch.quantization.quantize_dynamic(
        encoder, {torch.nn.Linear}, dtype=torch.qint8
    )
    if features.exists():
        X = np.load(features)
    else:
        chunks = []
        with torch.inference_mode():
            for i in range(0, len(d), batch_size):
                inputs = tok(
                    d.Text.iloc[i : i + batch_size].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=96,
                    return_tensors="pt",
                )
                hidden = encoder(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                chunks.append(pooled.numpy())
                if i % (batch_size * 20) == 0:
                    print("Encoded", i, "/", len(d), flush=True)
        X = np.vstack(chunks)
        np.save(features, X)
    del encoder
    y = d.Sentiment.to_numpy()
    tr = d.split.eq("train").to_numpy()
    va = d.split.eq("validation").to_numpy()
    te = d.split.eq("test").to_numpy()
    rows = []
    models = {}
    for c in [0.01, 0.1, 1, 10]:
        model = make_pipeline(
            StandardScaler(), LogisticRegression(C=c, max_iter=1500, random_state=SEED)
        )
        model.fit(X[tr], y[tr])
        name = f"Frozen CAMeLBERT + logistic C={c}"
        rows.append({"model": name, "C": c, **evaluate(y[va], model.predict(X[va]))})
        models[name] = model
    table = pd.DataFrame(rows)
    best = table.sort_values("macro_f1", ascending=False).iloc[0]["model"]
    model = models[best]
    dev = tr | va
    model.fit(X[dev], y[dev])
    pred = model.predict(X[te])
    scores = {
        **evaluate(y[te], pred),
        "accuracy_95pct_wilson": accuracy_interval(y[te], pred),
    }
    classification_artifacts(y[te], pred, out)
    table.to_csv(out / "validation_comparison.csv", index=False)
    compare_plot(table, "macro_f1", out / "validation_comparison.png")
    models_dir = out.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "classifier.joblib")
    report = {
        "dataset": "Same deduplicated sentiment data and partition as project 03",
        "input_sha256": sha256(data),
        "pretrained_encoder": model_name,
        "model_revision": revision,
        "encoder_training": "Frozen pretrained encoder; only logistic head trained on sentiment labels",
        "inference": "CPU dynamic int8 quantization of linear layers",
        "pooling": "attention-mask mean pooling, maximum 96 tokens",
        "selected_model": best,
        "selection_metric": "validation macro F1",
        "test_results": {best: scores},
        "split_sizes": d.split.value_counts().to_dict(),
        "runtime_seconds": time.time() - start,
        "environment": {**versions(), "torch": torch.__version__},
        "limitations": [
            "Dataset label provenance not independently verified.",
            "Pretraining corpus overlap cannot be independently excluded.",
            "Frozen features are not a fine-tuned transformer; truncation may lose long-text context.",
        ],
    }
    save_json(out / "metrics.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="results")
    p.add_argument("--model", default=MODEL)
    a = p.parse_args()
    run(a.data, a.out, a.model)
