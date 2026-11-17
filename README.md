# IDX Exchange: Home Price Prediction

Notebook source for the California single-family home price prediction internship project. Week 7 is `notebooks/week07_feature_importance.ipynb`; Week 8 is `notebooks/06_evaluation.ipynb`.

## Week 8 evaluation

Week 8 evaluates the saved Linear Regression, Decision Tree and Random Forest pipelines from Weeks 4–7. It calculates MAE, RMSE, R², MAPE, MdAPE and prediction bias, with price categories learned from training quantiles. April validation and May test results are labeled separately, as are all-cleaned and training-price-bounded populations. The April assessment is frozen before May predictions. May was examined in earlier project work and is not an untouched holdout.

## Run locally

Use the existing project Python environment, or install `requirements.txt` in a virtual environment. Execution requires authorized CRMLS files under `data/raw/`, the existing local district snapshot under `data/external/school_districts/`, and the saved Weeks 4–7 model artifacts and historical outputs. Those prerequisites are intentionally excluded from this repository; a fresh clone alone cannot reproduce the evaluation.

Open and execute `notebooks/06_evaluation.ipynb`, or run:

```powershell
.\.venv\Scripts\python.exe scripts\run_week8.py
.\.venv\Scripts\python.exe scripts\verify_week8.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Results are generated locally under `outputs/week08/`, including `metrics_summary.csv`, PNG/PDF figures, model inventory and verification evidence. The executed report is written to `docs/WEEK8_RESULTS.md`. Earlier project outputs are preserved.

## Data sharing

Raw or transformed MLS records, fitted models, predictions, licensed aggregate results, credentials and executed notebook outputs must remain local. Public notebooks have saved outputs removed. The ignore rules protect the data, external snapshots and generated outputs. Do not publish generated results reports without the applicable sharing authorization.
