"""IDX Exchange: leakage-aware California single-family AVM Week 1–4 workflow.

Run this against authorized CRMLS files downloaded locally by the intern. Raw
source data and generated outputs must not be committed to a public repository.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "ClosePrice"
DATE_COL = "CloseDate"
NUMERIC_CANDIDATES = [
    "LivingArea", "BedroomsTotal", "BathroomsTotalInteger",
    "LotSizeSquareFeet", "YearBuilt", "Latitude", "Longitude",
]
CATEGORICAL_CANDIDATES = ["PostalCode", "CountyOrParish"]
# Only intrinsic property attributes. NO list prices, DOM, closing-specific
# fields, price reductions, future market aggregates, or post-sale attributes.
FORBIDDEN_FEATURES = {
    "ListPrice", "OriginalListPrice", "DaysOnMarket", "CumulativeDaysOnMarket",
    "PriceChangeTimestamp", "ClosePrice", "CloseDate", "PurchaseContractDate",
    "ListPriceLow", "ListPriceHigh", "ContractStatusChangeDate",
}


def _json_dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf8")


def find_files(raw_dir: str | Path) -> list[Path]:
    p = Path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(f"Data directory missing: {p}. See data/raw/README.md")
    found = sorted(x for x in p.rglob("*") if x.is_file()
                   and x.name.lower().startswith("crmlssold")
                   and (x.name.lower().endswith(".csv") or
                        x.name.lower().endswith(".csv.gz") or
                        x.name.lower().endswith(".parquet")))
    if not found:
        raise FileNotFoundError("No CRMLSSold CSV/CSV.GZ/Parquet files under " + str(p))
    return found


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load files without assuming that one filename equals one calendar month."""
    files = find_files(raw_dir)
    frames: list[pd.DataFrame] = []
    for p in files:
        if p.suffix.lower() == ".parquet":
            frame = pd.read_parquet(p)
        else:
            frame = pd.read_csv(p, low_memory=False, dtype={"PostalCode": "string", "ListingKey": "string", "ParcelNumber": "string"})
        frame.columns = frame.columns.str.strip().str.lstrip("\ufeff")
        frame["_source_file"] = p.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), [str(p) for p in files]


def clean_properties(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove invalid/duplicate transactions, preserving auditable row counts."""
    essential = ["PropertyType", "PropertySubType", TARGET, DATE_COL]
    missing = [c for c in essential if c not in df]
    if missing:
        raise ValueError(f"Essential CRMLS columns missing: {missing}")
    out = df.copy()
    counts = {"raw": len(out)}

    # Dataset restriction mandated by the task prompt.
    out = out.loc[
        out["PropertyType"].astype("string").str.strip().eq("Residential")
        & out["PropertySubType"].astype("string").str.strip().eq("SingleFamilyResidence")
    ].copy()
    counts["single_family_residential"] = len(out)

    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce", utc=True).dt.tz_localize(None)
    out = out.loc[out[TARGET].notna() & (out[TARGET] > 0) & out[DATE_COL].notna()].copy()
    counts["positive_price_and_valid_close_date"] = len(out)

    if "LivingArea" in out:
        out["LivingArea"] = pd.to_numeric(out["LivingArea"], errors="coerce")
        out = out.loc[out["LivingArea"].isna() | (out["LivingArea"] > 0)].copy()
    counts["positive_or_missing_living_area"] = len(out)

    for c in ("BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.loc[out[c].isna() | (out[c] >= 0)].copy()
    counts["nonnegative_property_attributes"] = len(out)

    if "ListingContractDate" in out:
        listed = pd.to_datetime(out["ListingContractDate"], errors="coerce", utc=True).dt.tz_localize(None)
        out = out.loc[listed.isna() | (listed <= out[DATE_COL])].copy()
    counts["valid_transaction_timeline"] = len(out)

    # Exact duplicate rows can arise when the same file is imported twice.
    subset = [c for c in out if c != "_source_file"]
    out = out.drop_duplicates(subset=subset, keep="first")
    counts["no_exact_duplicates"] = len(out)

    # ListingKey is a unique record id, but if updates exist, use close date too.
    if "ListingKey" in out:
        known = out["ListingKey"].notna() & out["ListingKey"].astype("string").str.strip().ne("")
        unique = out.loc[known].drop_duplicates(subset=["ListingKey", DATE_COL], keep="last")
        out = pd.concat([unique, out.loc[~known]], ignore_index=True)
    counts["no_duplicate_listing_transactions"] = len(out)

    # Conservative: same parcel sold on a date is one transaction, but blanks
    # and absent parcel IDs must not collapse all records into one.
    if "ParcelNumber" in out:
        known = out["ParcelNumber"].notna() & out["ParcelNumber"].astype("string").str.strip().ne("")
        unique = out.loc[known].drop_duplicates(subset=["ParcelNumber", DATE_COL], keep="last")
        out = pd.concat([unique, out.loc[~known]], ignore_index=True)
    counts["deduplicated_transactions"] = len(out)
    if len(out) == 0:
        raise ValueError("No valid single-family home sales remain after cleaning.")
    return out.sort_values(DATE_COL).reset_index(drop=True), counts


def explore(df: pd.DataFrame, output_dir: str | Path = "outputs") -> dict[str, Any]:
    """Save EDA summary, missingness, price distribution and size/price plots."""
    out = Path(output_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    variables = [x for x in [TARGET, "LivingArea", "BedroomsTotal", "BathroomsTotalInteger", "LotSizeSquareFeet"] if x in df]
    summary = df[variables].apply(pd.to_numeric, errors="coerce").describe().round(2)
    summary.to_csv(out / "eda_descriptive_statistics.csv")
    missing = pd.DataFrame({"missing_count": df.isna().sum(), "missing_percentage": (df.isna().mean()*100).round(2)})
    missing.sort_values("missing_percentage", ascending=False).to_csv(out / "eda_missingness.csv")
    for c in variables:
        values = pd.to_numeric(df[c], errors="coerce").dropna()
        if values.empty:
            continue
        # Trim plots ONLY, not the analysis, to keep extreme prices legible.
        lo, hi = values.quantile([.01, .99])
        shown = values.loc[values.between(lo, hi)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(shown, bins=50, edgecolor="white", linewidth=.3)
        ax.set(title=f"{c}: middle 98% of observations", xlabel=c, ylabel="Properties")
        fig.tight_layout()
        fig.savefig(figs / f"{c}_distribution.png", dpi=160)
        plt.close(fig)
    if {"LivingArea", TARGET}.issubset(df.columns):
        scatter = df[["LivingArea",TARGET]].apply(pd.to_numeric,errors="coerce").dropna()
        if len(scatter) > 50000: scatter = scatter.sample(50000, random_state=42)
        if len(scatter) > 1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.scatter(scatter["LivingArea"], scatter[TARGET], s=4, alpha=.2)
            ax.set(title="Living area vs. close price", xlabel="Living area (sq ft)", ylabel="Close price ($)")
            fig.tight_layout()
            fig.savefig(figs / "living_area_vs_close_price.png", dpi=160)
            plt.close(fig)
    months = df[DATE_COL].dt.to_period("M").astype(str).value_counts().sort_index()
    result = {
        "rows": len(df), "columns": len(df.columns),
        "sale_date_range": [str(df[DATE_COL].min().date()), str(df[DATE_COL].max().date())],
        "months_available": months.to_dict(),
        "close_price_median": float(df[TARGET].median()),
        "close_price_mean": float(df[TARGET].mean()),
        "note": "These are real-data results only when actual authorized CRMLSSold files are supplied.",
    }
    _json_dump(out / "eda_summary.json", result)
    return result


def chronological_split(df: pd.DataFrame, as_of: str | date | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Latest complete month = test; second most recent = validation; earlier = train."""
    cutoff = pd.Timestamp(as_of).to_period("M") if as_of else pd.Timestamp.today().to_period("M")
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    period = df[DATE_COL].dt.to_period("M")
    valid = df.loc[(period < cutoff)].copy()
    months = sorted(valid[DATE_COL].dt.to_period("M").unique())
    if len(months) < 3:
        raise ValueError(f"Need >=3 complete months (train/validation/test); found {len(months)}. Download >=6.")
    test_month, val_month = months[-1], months[-2]
    period = valid[DATE_COL].dt.to_period("M")
    # Retain all older months; do not randomly shuffle or train on future rows.
    chunks = {
        "train": valid.loc[period < val_month].copy(),
        "validation": valid.loc[period == val_month].copy(),
        "test": valid.loc[period == test_month].copy(),
    }
    if any(v.empty for v in chunks.values()):
        raise ValueError("At least one chronological split is empty.")
    detail = {
        "as_of": str(cutoff), "available_complete_months": [str(x) for x in months],
        "validation_month": str(val_month), "test_month": str(test_month),
        "sizes_before_outlier_filter": {k: len(v) for k,v in chunks.items()},
        "incomplete_or_future_rows_excluded": int(len(df)-len(valid)),
    }
    return chunks, detail


def price_outlier_filter(chunks: dict[str, pd.DataFrame]) -> tuple[dict[str,pd.DataFrame],dict[str,Any]]:
    """Freeze 0.5/99.5 percentiles from training targets only per AVM manual."""
    train_price = chunks["train"][TARGET]
    low, high = (float(x) for x in train_price.quantile([0.005, 0.995]))
    filtered = {name: frame.loc[frame[TARGET].between(low,high)].copy() for name,frame in chunks.items()}
    if any(v.empty for v in filtered.values()):
        raise ValueError("A chronological partition became empty after the training-derived price filter.")
    return filtered, {"train_price_min":low,"train_price_max":high,
                      "rows_after_outlier_filter":{k:len(v) for k,v in filtered.items()},
                      "warning":"Metrics describe sales within training-derived target-price bounds; retain an unfiltered sensitivity check for later work."}


def choose_features(train: pd.DataFrame) -> tuple[list[str],list[str]]:
    num = [c for c in NUMERIC_CANDIDATES if c in train and pd.to_numeric(train[c],errors="coerce").notna().any()]
    cat = [c for c in CATEGORICAL_CANDIDATES if c in train and train[c].notna().any()]
    if not num:
        raise ValueError("No usable numeric property characteristics exist in training data.")
    if any(x in FORBIDDEN_FEATURES for x in num+cat):
        raise AssertionError("An excluded/leaky feature was selected.")
    return num,cat


def build_pipeline(numeric: list[str], categorical: list[str]) -> Pipeline:
    numpipe = Pipeline([
        ("impute", SimpleImputer(strategy="median",add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    transformers = [("numeric", numpipe, numeric)]
    if categorical:
        catpipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore",max_categories=100, min_frequency=25)),
        ])
        transformers.append(("categorical", catpipe, categorical))
    pre = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=.3)
    return Pipeline([("preprocess",pre), ("model",LinearRegression())])


def metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str,float]:
    true = np.asarray(y_true,dtype=float)
    pred = np.asarray(pred,dtype=float)
    ape = np.abs(true-pred)/true
    return {"R2": float(r2_score(true,pred)), "MAE_USD":float(mean_absolute_error(true,pred)),
            "MAPE_pct":float(mean_absolute_percentage_error(true,pred)*100),
            "MdAPE_pct":float(np.median(ape)*100)}


def train_baseline(chunks: dict[str,pd.DataFrame], output_dir: str|Path="outputs",
                   evaluate_test: bool=False) -> dict[str,Any]:
    """Fit on train only. Only evaluate official test set when explicitly requested."""
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    num,cat = choose_features(chunks["train"])
    features = num+cat
    pipeline = build_pipeline(num,cat)
    Xtr = chunks["train"][features].copy()
    Xtr[num] = Xtr[num].apply(pd.to_numeric,errors="coerce")
    for c in cat:
        Xtr[c] = Xtr[c].astype(object).where(Xtr[c].notna(), np.nan)
    pipeline.fit(Xtr,chunks["train"][TARGET])
    report={"model":"scikit-learn LinearRegression", "numeric_features":num,
            "categorical_features":cat,
            "warning":"No data leakage: preprocessing learned from train only. Test evaluated only when explicitly requested."}
    for name in ("train","validation","test"):
        if name == "test" and not evaluate_test: continue
        frame=chunks[name]
        x=frame[features].copy()
        x[num]=x[num].apply(pd.to_numeric,errors="coerce")
        for c in cat:
            x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
        pred=pipeline.predict(x)
        report[name]=metrics(frame[TARGET],pred)
        if name in ("validation","test"):
            pd.DataFrame({"actual":frame[TARGET].values,"predicted":pred,
                          "absolute_percentage_error":np.abs(frame[TARGET].values-pred)/frame[TARGET].values
                          }).to_csv(out/f"{name}_prediction_errors.csv",index=False)
    joblib.dump({"pipeline":pipeline,"features":features,"numeric_features":num,
                 "categorical_features":cat}, out/"baseline_pipeline.joblib")
    _json_dump(out/"baseline_metrics.json",report)
    return report


def write_clean_local_csv(df:pd.DataFrame,output_dir:str|Path) -> Path:
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    # Contains licensed transactions: intentionally generated only locally, ignored by git.
    path=out/"california_single_family_cleaned.csv"
    # Minimize copied MLS fields: do not export unrelated address, access,
    # contact, or agent metadata that would expose sensitive MLS information.
    keep=["PropertyType", "PropertySubType", DATE_COL, TARGET]
    keep += NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES
    keep=list(dict.fromkeys(c for c in keep if c in df.columns))
    df[keep].to_csv(path,index=False)
    return path


def run(raw_dir:str|Path="data/raw",output_dir:str|Path="outputs",as_of:str|None=None,
        through:str="train",evaluate_test:bool=False) -> dict[str,Any]:
    """Entrypoint for notebooks and CLI. Exact same functions used in both."""
    raw, files = load_raw(raw_dir)
    cleaned, counts=clean_properties(raw)
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    stages=list(counts.items())
    per_stage={}
    for i,(stage,remaining) in enumerate(stages):
        previous=stages[i-1][1] if i else remaining
        removed=int(previous-remaining)
        per_stage[stage]={"removed":removed,
                           "pct_removed_of_previous":round(100*removed/previous,2) if previous else 0.0}
    audit={"input_files":[Path(x).name for x in files],
           "source_file_bytes":{Path(x).name:Path(x).stat().st_size for x in files},
           "row_counts":counts,"removed_by_stage":per_stage,
           "total_removed":int(counts["raw"]-len(cleaned))}
    _json_dump(output/"data_cleaning_audit.json",audit)
    result={"cleaning":audit}
    if through in ("explore","preprocess","train"):
        result["exploration"]=explore(cleaned,output)
    if through in ("preprocess","train"):
        csv=write_clean_local_csv(cleaned,output)
        chunks,split=chronological_split(cleaned,as_of)
        chunks,outliers=price_outlier_filter(chunks)
        result["data_split"]={**split,"outlier_filter":outliers}
        _json_dump(output/"data_split_audit.json",result["data_split"])
        result["clean_csv_path"]=str(csv)
    if through=="train":
        result["model"]=train_baseline(chunks,output,evaluate_test=evaluate_test)
    _json_dump(output/"project_run_summary.json",result)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description="California CRMLS single-family AVM (Weeks 1–4)")
    p.add_argument("--data-dir",default="data/raw")
    p.add_argument("--output-dir",default="outputs")
    p.add_argument("--as-of",default=None,help="YYYY-MM-DD; rows in the current month are excluded")
    p.add_argument("--through",choices=["explore","preprocess","train"],default="train")
    p.add_argument("--final-test",action="store_true",help="Evaluate official test month once, after model frozen")
    args=p.parse_args()
    report=run(args.data_dir,args.output_dir,args.as_of,args.through,args.final_test)
    print(json.dumps(report,indent=2,default=str))

if __name__=="__main__": main()
