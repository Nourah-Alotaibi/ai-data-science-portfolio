"""Layoff EDA that reports observed totals without imputing missing events."""

import argparse
from pathlib import Path
import pandas as pd, numpy as np
from common import *


def run(data, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data)
    n = len(df)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    for c in df.select_dtypes("object"):
        df[c] = (
            df[c]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "Unknown": pd.NA, "N/A": pd.NA})
        )
    df["date"] = pd.to_datetime(df.Date_layoffs, errors="coerce")
    df["Laid_Off"] = pd.to_numeric(df.Laid_Off, errors="coerce")
    negative = int((df.Laid_Off < 0).sum())
    df.loc[df.Laid_Off < 0, "Laid_Off"] = np.nan
    missing = df.isna().sum().sort_values(ascending=False)
    missing.rename("missing_count").to_csv(out / "missingness.csv")
    df["month"] = df.date.dt.to_period("M").astype("string")
    df["year"] = df.date.dt.year
    monthly = (
        df[df.date.notna()]
        .groupby("month")
        .agg(
            observed_layoffs=("Laid_Off", lambda s: s.sum(min_count=1)),
            events=("Company", "size"),
            events_with_counts=("Laid_Off", "count"),
        )
    )
    monthly.to_csv(out / "monthly_observed_layoffs.csv")
    annual = df.groupby("year").agg(
        observed_layoffs=("Laid_Off", lambda s: s.sum(min_count=1)),
        events=("Company", "size"),
    )
    annual.to_csv(out / "annual_observed_layoffs.csv")
    industry = (
        df.groupby("Industry", dropna=False)
        .Laid_Off.sum(min_count=1)
        .sort_values(ascending=False)
    )
    industry.to_csv(out / "industry_observed_layoffs.csv")
    country = (
        df.groupby("Country", dropna=False)
        .Laid_Off.sum(min_count=1)
        .sort_values(ascending=False)
    )
    country.to_csv(out / "country_observed_layoffs.csv")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        pd.to_datetime(monthly.index),
        monthly.observed_layoffs,
        marker="o",
        markersize=3,
        color="#435fa5",
    )
    ax.set_title("Reported layoffs by month — observed counts only")
    ax.set_ylabel("Reported employees laid off")
    ax.set_xlabel("Month")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out / "monthly_layoffs.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    industry.head(10).sort_values().plot.barh(ax=ax, color="#7864af")
    ax.set_xlabel("Reported employees laid off")
    ax.set_title("Industries with the largest observed totals")
    fig.tight_layout()
    fig.savefig(out / "industry_layoffs.png")
    plt.close(fig)
    report = {
        "dataset": "Local tech_layoffs_til_2025.csv snapshot",
        "input_sha256": sha256(data),
        "source_rows": n,
        "clean_rows": len(df),
        "exact_duplicates_removed": duplicates,
        "negative_counts_set_missing": negative,
        "missing_layoff_count_rows": int(df.Laid_Off.isna().sum()),
        "invalid_or_missing_dates": int(df.date.isna().sum()),
        "observed_total_layoffs": float(df.Laid_Off.sum()),
        "start_date": str(df.date.min().date()),
        "end_date": str(df.date.max().date()),
        "largest_reported_industry": str(industry.index[0]),
        "largest_reported_industry_total": float(industry.iloc[0]),
        "limitations": [
            "Reported records are not a census of all layoffs.",
            "Missing counts are not imputed as zero or median.",
            "Partial calendar periods are not compared as full years.",
            "Counts do not establish why layoffs happened.",
        ],
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
