"""Local reproduction checks for the frozen baseline; no model selection."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import avm


def verify_frozen_baseline(splits, bounded_splits, output_dir):
    out = Path(output_dir)
    artifact = joblib.load(out / "baseline_pipeline.joblib")
    model = artifact["pipeline"]
    features = artifact["features"]
    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]

    def predict(frame):
        x = frame[features].copy()
        x[numeric] = x[numeric].apply(pd.to_numeric, errors="coerce")
        for col in categorical:
            x[col] = x[col].astype(object).where(x[col].notna(), np.nan)
        return model.predict(x)

    report = json.loads((out / "baseline_metrics.json").read_text(encoding="utf-8"))
    for name, frame in bounded_splits.items():
        report[name] = avm.metrics(frame[avm.TARGET], predict(frame))
    avm._json_dump(out / "baseline_metrics.json", report)
    summary = pd.DataFrame({name: report[name] for name in bounded_splits}).T
    summary.index.name = "split"
    summary.to_csv(out / "metrics_summary.csv")

    test = splits["test"]
    sensitivity = {"months": str(test[avm.DATE_COL].dt.to_period("M").iloc[0]),
                   "rows": len(test), "metrics": avm.metrics(test[avm.TARGET], predict(test)),
                   "bounded_rows": len(bounded_splits["test"]),
                   "note": "Same frozen fitted model; reproduction only, never used for tuning."}
    avm._json_dump(out / "unfiltered_test_sensitivity.json", sensitivity)
    frame = bounded_splits["test"]
    actual = frame[avm.TARGET].to_numpy()
    error = np.abs(actual - predict(frame))
    bands = pd.cut(actual, [0, 500000, 1000000, 2000000, 5000000, np.inf],
                   right=False, labels=["<$500k", "$500k–$1m", "$1m–$2m", "$2m–$5m", "≥$5m"])
    grouped = pd.DataFrame({"price_band": bands, "APE": error / actual * 100,
                            "error": error}).groupby("price_band", observed=False)
    band_metrics = grouped.agg(n=("APE", "size"), median_APE_pct=("APE", "median"),
                               mean_APE_pct=("APE", "mean"), mean_abs_error_usd=("error", "mean"))
    band_metrics.round(2).to_csv(out / "price_band_metrics.csv")
    return summary, band_metrics, sensitivity
