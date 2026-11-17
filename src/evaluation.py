"""Consistent, auditable regression metrics and training-derived price bands.

Percentage errors are percentages (100 * |prediction - actual| / actual).
Nonfinite actuals are counted and excluded; finite nonpositive actuals remain
in dollar metrics but are excluded from percentage metrics. Invalid predictions
raise rather than quietly discarding difficult rows.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BAND_NAMES = ['Lower-priced', 'Mid-range', 'Upper-mid-range', 'Luxury']
METRICS = ['MAE_USD', 'RMSE_USD', 'R2', 'MAPE_pct', 'MdAPE_pct']


def regression_metrics(actual, predicted, minimum_r2_n=2):
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or y.shape != p.shape:
        raise ValueError('Actual and predicted must be aligned one-dimensional arrays')
    if minimum_r2_n < 2:
        raise ValueError('R2 requires at least two observations')
    if not np.isfinite(p).all():
        raise ValueError('Nonfinite predictions: evaluation must not silently discard rows')
    valid = np.isfinite(y)
    y, p = y[valid], p[valid]
    positive = y > 0
    result = dict(n_total=len(valid), n=len(y), n_invalid_actual=int((~valid).sum()),
                  n_percentage=int(positive.sum()), n_excluded_percentage=int((~positive).sum()))
    result.update({key: np.nan for key in METRICS + ['bias_USD']})
    if not len(y):
        result['R2_reason'] = 'no_valid_actuals'
        return result
    residual = p-y
    result.update(MAE_USD=float(mean_absolute_error(y, p)),
                  RMSE_USD=float(np.sqrt(mean_squared_error(y, p))),
                  bias_USD=float(residual.mean()))
    if positive.any():
        ape = np.abs(residual[positive])/y[positive]*100
        result.update(MAPE_pct=float(ape.mean()), MdAPE_pct=float(np.median(ape)))
    if len(y) < minimum_r2_n:
        result['R2_reason'] = 'insufficient_n'
    elif np.all(y == y[0]):
        result['R2_reason'] = 'constant_actuals'
    else:
        result.update(R2=float(r2_score(y, p, force_finite=False)), R2_reason='reported')
    return result


def training_price_bands(prices):
    """Use full cleaned training targets only: Q25, Q50, and Q90 boundaries.

    Lower 25%, next 25%, next 40%, top 10% (approximately, due to ties).
    Luxury is a relative top-decile label, not an external market definition.
    """
    y = np.asarray(prices, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all() or (y <= 0).any():
        raise ValueError('Band fitting requires nonempty, finite positive training prices')
    quantiles = [.0, .005, .01, .25, .5, .75, .9, .95, .99, .995, 1.]
    values = np.quantile(y, quantiles, method='linear')
    cuts = np.quantile(y, [.25, .5, .9], method='linear').tolist()
    if not np.all(np.diff(cuts) > 0):
        raise ValueError('Training quantiles collapse; four distinct bands are unavailable')
    return dict(labels=BAND_NAMES, cuts=cuts, training_n=len(y),
                quantiles={str(q): float(v) for q, v in zip(quantiles, values)},
                source='All cleaned December 2025-March 2026 training prices only',
                intervals='lower inclusive, upper exclusive; unbounded outer intervals',
                quantile_method='linear', probabilities=[.25, .5, .9])


def assign_price_bands(actual, specification):
    return pd.cut(np.asarray(actual, dtype=float),
                  [-np.inf] + specification['cuts'] + [np.inf],
                  labels=specification['labels'], right=False)


def metric_rows(actual, predicted, specification):
    """One overall row plus all four bands, including empty categories."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    rows = [dict(price_band='All', **regression_metrics(y, p))]
    bands = assign_price_bands(y, specification)
    for label in specification['labels']:
        mask = bands == label
        rows.append(dict(price_band=label,
                         **regression_metrics(y[mask], p[mask], minimum_r2_n=50)))
    return rows


def validation_assessment(summary):
    """April overall rows only; all-cleaned is primary, bounded is secondary.

    Equal rank weight on R2, MAE, MdAPE avoids counting R2 and RMSE twice.
    Ranks identify tradeoffs, not meaningful/significant differences.
    """
    validation = summary.loc[(summary.split == 'validation') & (summary.price_band == 'All')]
    if validation.empty:
        raise ValueError('April validation metrics required for model assessment')
    rows, leaders = [], {}
    for population in ('all_cleaned', 'bounded'):
        table = validation.loc[validation.population == population].copy()
        if table.empty or table.model_id.duplicated().any():
            raise ValueError('Expected one validation overall row per model and population')
        if not np.isfinite(table[['R2', 'MAE_USD', 'MdAPE_pct']]).all().all():
            raise ValueError('Cannot rank undefined validation metrics')
        for metric, ascending in [('R2', False), ('MAE_USD', True), ('MdAPE_pct', True)]:
            table[f'rank_{metric}'] = table[metric].rank(ascending=ascending, method='average')
        table['balanced_rank'] = table[['rank_R2', 'rank_MAE_USD', 'rank_MdAPE_pct']].mean(axis=1)
        ordered = table.sort_values(['balanced_rank', 'model_id'], kind='stable')
        leaders[population] = {
            metric: sorted(table.loc[np.isclose(table[metric],
                table[metric].max() if metric == 'R2' else table[metric].min(), rtol=0, atol=0),
                'model_id'].tolist()) for metric in ('R2', 'MAE_USD', 'MdAPE_pct')}
        leaders[population]['balanced'] = ordered.iloc[0].model_id
        rows.append(table)
    return pd.concat(rows, ignore_index=True), dict(
        selection_split='validation', primary_population='all_cleaned',
        rule='Equal mean ranks of R2 (descending), MAE and MdAPE (ascending); model_id breaks rank ties',
        leaders=leaders, selected_model_id=leaders['all_cleaned']['balanced'],
        may_used_for_selection=False)
