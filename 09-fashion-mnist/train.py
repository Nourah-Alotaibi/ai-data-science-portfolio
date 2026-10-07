"""Reproducible Fashion-MNIST baseline and CNN with validation checkpoints."""

import argparse, copy, gzip, hashlib, struct, time, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from common import *

FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
LABELS = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load(data):
    data = Path(data)
    data.mkdir(parents=True, exist_ok=True)
    arr = []
    for name, md5 in FILES.items():
        p = data / name
        if not p.exists():
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
                + name,
                p,
            )
        if hashlib.md5(p.read_bytes()).hexdigest() != md5:
            raise ValueError("Dataset checksum mismatch: " + name)
        b = gzip.decompress(p.read_bytes())
        magic = struct.unpack(">I", b[:4])[0]
        a = np.frombuffer(b, dtype=np.uint8, offset=16 if magic == 2051 else 8).copy()
        arr.append(a.reshape(-1, 1, 28, 28) if magic == 2051 else a)
    return arr


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.15),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


def run(data, out, epochs=10):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    X, y, Xt, yt = load(data)
    tr, va = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED
    )
    # Normalize using training pixels only. Official test set never selects epochs.
    mean = float(X[tr].mean() / 255)
    std = float(X[tr].std() / 255)

    def ds(a, b):
        return TensorDataset(
            (torch.tensor(a, dtype=torch.float32) / 255 - mean) / std,
            torch.tensor(b, dtype=torch.long),
        )

    dev = ds(X, y)
    te = ds(Xt, yt)
    train = DataLoader(
        torch.utils.data.Subset(dev, tr),
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    val = DataLoader(torch.utils.data.Subset(dev, va), batch_size=512)
    test = DataLoader(te, batch_size=512)

    def predict(m, loader):
        m.eval()
        pp = []
        yy = []
        with torch.inference_mode():
            for a, b in loader:
                pp.extend(m(a).argmax(1).numpy())
                yy.extend(b.numpy())
        return np.asarray(yy), np.asarray(pp)

    candidates = {
        "MLP baseline": (
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 10),
            ),
            5,
        ),
        "Regularized CNN": (CNN(), epochs),
    }
    history = []
    states = {}
    validation = []
    for name, (m, n_epoch) in candidates.items():
        opt = torch.optim.AdamW(m.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epoch)
        best = -1
        for epoch in range(n_epoch):
            m.train()
            losses = []
            for a, b in train:
                opt.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(m(a), b)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            scheduler.step()
            vv, pred = predict(m, val)
            score = evaluate(vv, pred)
            history.append(
                {
                    "model": name,
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    **score,
                }
            )
            print(name, epoch + 1, score, flush=True)
            if score["accuracy"] > best:
                best = score["accuracy"]
                states[name] = copy.deepcopy(m.state_dict())
        m.load_state_dict(states[name])
        vv, pred = predict(m, val)
        validation.append({"model": name, **evaluate(vv, pred)})
    table = pd.DataFrame(validation)
    selected = table.sort_values("accuracy", ascending=False).iloc[0]["model"]
    tests = {}
    # Evaluate both pre-specified candidates once; winner is fixed by validation.
    for name, (m, _) in candidates.items():
        yy, pred = predict(m, test)
        tests[name] = {
            **evaluate(yy, pred),
            "accuracy_95pct_wilson": accuracy_interval(yy, pred),
        }
        if name == selected:
            classification_artifacts(yy, pred, out, labels=LABELS)
            models = out.parent / "models"
            models.mkdir(exist_ok=True)
            torch.save(
                {
                    "state_dict": m.state_dict(),
                    "mean": mean,
                    "std": std,
                    "architecture": name,
                },
                models / "model.pt",
            )
    h = pd.DataFrame(history)
    h.to_csv(out / "learning_history.csv", index=False)
    table.to_csv(out / "validation_comparison.csv", index=False)
    fig, ax = plt.subplots()
    for name, g in h.groupby("model"):
        ax.plot(g.epoch, g.accuracy, marker="o", label=name)
    ax.set(
        xlabel="Epoch",
        ylabel="Validation accuracy",
        title="Fashion-MNIST validation learning curves",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png")
    plt.close(fig)
    report = {
        "dataset": "Official Zalando Fashion-MNIST, checksums verified",
        "split_sizes": {"train": len(tr), "validation": len(va), "test": len(yt)},
        "selected_model": selected,
        "selection_metric": "validation accuracy",
        "test_results": tests,
        "normalization": {"mean": mean, "std": std},
        "runtime_seconds": time.time() - start,
        "environment": {**versions(), "torch": torch.__version__},
        "limitations": [
            "Small benchmark images; not evaluated on real product photographs.",
            "CNN and MLP have different fixed training budgets; not a compute-matched comparison.",
        ],
    }
    save_json(out / "metrics.json", report)
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--epochs", type=int, default=10)
    a = p.parse_args()
    run(a.data, a.out, a.epochs)
