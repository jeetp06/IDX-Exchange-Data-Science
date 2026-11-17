"""Static Week 8 figures and findings, derived exclusively from executed results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.evaluation import assign_price_bands, BAND_NAMES

CHARTS = ['actual_vs_predicted','residual_distribution','ape_distribution',
          'mdape_by_price_band','mae_by_price_band','model_comparison','bias_by_price_band']
COLORS = ['#176b87','#df8455','#5a8c63']


def short_name(name):
    return name.replace('Linear Regression','LR').replace('Decision Tree','DT').replace(
        'Random Forest','RF').replace('without_property_transformations','no property transforms')


def save_figure(fig, directory, name):
    fig.savefig(directory/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(directory/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)


def write_figures(root, summary, specs, selection, bands):
    out = root/'outputs/week08'
    directory = out/'figures'; directory.mkdir(exist_ok=True)
    selected = selection['selected_model_id']
    selected_index = next(i for i,s in enumerate(specs) if s['model_id']==selected)
    # Historical comparator plus reference makes the feature-removal change visible.
    shown = list(dict.fromkeys(['W04 Linear Regression / original',
                 'W06 Random Forest / engineered_districts',selected]))
    style = {'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
             'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
             'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
             'pdf.fonttype':42}
    money = FuncFormatter(lambda x,_: f'${x/1e6:g}m' if abs(x)>=1e6 else f'${x/1e3:g}k')
    with plt.rc_context(style):
        # Training distribution is examined in full before evaluation categories.
        private = np.load(out/'predictions_private'/f'model_{selected_index:02d}_train.npz')
        prices = private['actual']
        fig,ax = plt.subplots(figsize=(10,5),layout='constrained')
        ax.hist(prices,bins=np.geomspace(prices.min(),prices.max(),65),color=COLORS[0],edgecolor='white',linewidth=.3)
        for q,cut,color in zip(['Q25','Q50','Q90'],bands['cuts'],['#bd4b42','#7a5195','#b08d1c']):
            ax.axvline(cut,color=color,linewidth=1.5,label=f'{q}: ${cut:,.0f}')
        ax.set(xscale='log',xlabel='Recorded sale price (USD; logarithmic scale)',ylabel='Training sales per bin',
               title=f'Full cleaned training price distribution (n={len(prices):,})')
        ax.legend(); ax.grid(axis='y',alpha=.18)
        save_figure(fig,directory,'training_price_distribution')
        for split in ('validation','test'):
            data = np.load(out/'predictions_private'/f'model_{selected_index:02d}_{split}.npz')
            for population in ('all_cleaned','bounded'):
                mask = np.ones(len(data['actual']),dtype=bool) if population=='all_cleaned' else data['bounded']
                y,p = data['actual'][mask],data['predicted'][mask]
                residual = p-y; ape = np.abs(residual)/y*100
                prefix = f'{split}_{population}_'
                context = ('April validation' if split=='validation' else 'May test (previously examined)')
                context += f' | {population.replace("_"," ")} | n={len(y):,}'
                subtitle = short_name(selected)

                fig,ax = plt.subplots(figsize=(8,7),layout='constrained')
                ax.scatter(y,p,s=5,alpha=.3,color=COLORS[0],rasterized=True,label='All evaluated sales')
                lo,hi = min(y.min(),p.min()),max(y.max(),p.max())
                ax.plot([lo,hi],[lo,hi],ls='--',color='#b34d44',label='Perfect prediction')
                ax.set(xlabel='Actual sale price (USD)',ylabel='Predicted price (USD)',
                       title='Actual vs. predicted prices\n'+context+'\n'+subtitle)
                ax.set_xscale('symlog',linthresh=10000)
                ax.set_yscale('symlog',linthresh=10000)
                ax.xaxis.set_major_formatter(money); ax.yaxis.set_major_formatter(money)
                ax.legend(loc='upper left'); ax.grid(alpha=.15)
                fig.supxlabel('Symmetric-log axes retain negative predictions; all points shown',fontsize=9)
                save_figure(fig,directory,prefix+'actual_vs_predicted')

                fig,ax = plt.subplots(figsize=(10,5),layout='constrained')
                extent = max(float(np.abs(residual).max()),10001.)
                edges = np.unique(np.r_[-np.geomspace(extent,10000,40),np.linspace(-10000,10000,21),
                                        np.geomspace(10000,extent,40)])
                ax.hist(residual,bins=edges,color=COLORS[0],edgecolor='white',linewidth=.3)
                ax.axvline(0,color='#b34d44',ls='--',label='Zero error')
                ax.axvline(residual.mean(),color='#df8455',label=f'Mean bias: ${residual.mean():,.0f}')
                ax.set_xscale('symlog',linthresh=10000)
                ax.xaxis.set_major_formatter(money)
                ax.set(xlabel='Predicted minus actual (USD; symmetric-log scale)',ylabel='Sales per bin',
                       title='Signed residual distribution\n'+context+'\n'+subtitle)
                ax.legend(); ax.grid(axis='y',alpha=.18)
                fig.supxlabel('All residuals retained; bins have unequal dollar widths',fontsize=9)
                save_figure(fig,directory,prefix+'residual_distribution')

                fig,ax = plt.subplots(figsize=(10,5),layout='constrained')
                edges = np.expm1(np.linspace(0,np.log1p(max(ape.max(),1)),65))
                edges[-1] = max(edges[-1],ape.max())
                ax.hist(ape,bins=edges,color=COLORS[0],edgecolor='white',linewidth=.3)
                ax.axvline(np.median(ape),color='#df8455',label=f'MdAPE: {np.median(ape):.2f}%')
                ax.set_xscale('symlog',linthresh=1)
                ax.set(xlabel='Absolute percentage error (%; symmetric-log scale)',ylabel='Sales per bin',
                       title='Absolute percentage error distribution\n'+context+'\n'+subtitle)
                ax.legend(); ax.grid(axis='y',alpha=.18)
                fig.supxlabel('All errors retained; positive actual prices only; unequal percentage-bin widths',fontsize=9)
                save_figure(fig,directory,prefix+'ape_distribution')

                subset = summary.loc[(summary.split==split)&(summary.population==population)]
                for metric,name,label in [('MdAPE_pct','mdape_by_price_band','Median absolute percentage error (%)'),
                                          ('MAE_USD','mae_by_price_band','Mean absolute error (USD)'),
                                          ('bias_USD','bias_by_price_band','Mean predicted minus actual (USD)')]:
                    fig,ax = plt.subplots(figsize=(12,6),layout='constrained')
                    positions = np.arange(4); width=.8/len(shown)
                    for i,model in enumerate(shown):
                        table = subset.loc[(subset.model_id==model)&(subset.price_band!='All')].set_index('price_band').reindex(BAND_NAMES)
                        ax.bar(positions-.4+width*(i+.5),table[metric],width,color=COLORS[i],label=short_name(model))
                    counts = subset.loc[(subset.model_id==selected)&(subset.price_band!='All')].set_index('price_band').reindex(BAND_NAMES).n
                    ax.set_xticks(positions,[f'{b}\nn={int(counts[b]):,}' for b in BAND_NAMES])
                    if metric!='MdAPE_pct': ax.yaxis.set_major_formatter(money)
                    if metric=='bias_USD':
                        ax.axhline(0,color='#444444',linewidth=.8)
                        ax.set_yscale('symlog',linthresh=10000)
                        label += '; symmetric-log scale'
                    ax.set(ylabel=label,xlabel='Actual-price category (fixed training quantiles)',
                           title=label.split(';')[0]+' by price category\n'+context)
                    ax.legend(fontsize=9); ax.grid(axis='y',alpha=.18)
                    save_figure(fig,directory,prefix+name)

                overall = subset.loc[subset.price_band=='All'].sort_values('model_id')
                fig,axes = plt.subplots(2,3,figsize=(21,12),layout='constrained')
                labels = [short_name(v) for v in overall.model_id]
                for ax,metric,title in zip(axes.flat,['R2','MAE_USD','RMSE_USD','MAPE_pct','MdAPE_pct'],
                                           ['R2 (higher is better)','MAE USD (lower is better)',
                                            'RMSE USD (lower is better)','MAPE % (lower is better)',
                                            'MdAPE % (lower is better)']):
                    colors = ['#df8455' if v==selected else '#176b87' for v in overall.model_id]
                    ax.barh(labels,overall[metric],color=colors)
                    ax.invert_yaxis(); ax.set(xlabel=title); ax.tick_params(axis='y',labelsize=8)
                    ax.grid(axis='x',alpha=.18)
                    if metric.endswith('USD'): ax.xaxis.set_major_formatter(money)
                axes.flat[-1].axis('off')
                axes.flat[-1].text(.05,.5,'Orange: April all-cleaned balanced-rank choice\nBlue: other frozen models\n\nLR = Linear Regression\nDT = Decision Tree\nRF = Random Forest\n\nNo models refitted; no May-based selection.',va='center',fontsize=12)
                fig.suptitle('Frozen model comparison\n'+context,fontsize=16)
                save_figure(fig,directory,prefix+'model_comparison')


def markdown(table):
    view = table.copy()
    for column in view.select_dtypes(include='number'):
        view[column] = view[column].map(lambda v: 'undefined' if pd.isna(v) else f'{v:,.6f}')
    return '| '+' | '.join(view.columns)+' |\n| '+' | '.join(['---']*len(view.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(map(str,row))+' |' for row in view.itertuples(index=False,name=None))


def write_report(root, summary, inventory, manifest):
    out = root/'outputs/week08'
    selection=manifest['selection']; selected=selection['selected_model_id']
    cuts = manifest['price_bands']['cuts']
    text = f'''# Week 8: comprehensive model evaluation and price-band analysis

This report was generated from executed predictions of {len(inventory)} saved pipelines. No model was fitted or tuned in Week 8. All figures and metrics are local, licensed project artifacts. May was examined previously and is not an untouched holdout.

## Repository audit and verified prior work

Weeks 4-6 provide Linear Regression, Decision Tree and Random Forest pipelines. Week 5 has seven candidate fits but only three saved family winners. Week 6 adds six saved engineered/district pipelines and reuses the three Week 5 originals. Week 7 has six removal experiments, with its April winner (`without_property_transformations`) persisted; some other removals are equivalent to saved Week 5/6 pipelines. No Gradient Boosting, XGBoost or LightGBM implementation or artifact was found. Metrics-only candidates are listed in `outputs/week08/archived_experiments.csv`; they are not presented as newly evaluated pipelines. The separate Week 4 and Week 5 linear artifacts are retained and labeled explicitly.

Week 7 artifact verification and the prior test suite passed before implementation. This evaluation reproduced {manifest['historical_reproduction_checks']} historical metric values from the loaded models on their bounded training/validation populations (and baseline test where recorded). Differences and expected values are in `historical_reproduction.csv`. All prior weekly files are hash-checked unchanged. Historical verification scripts that overwrite old results were intentionally not rerun.

{markdown(inventory[['model_id','estimator','predictors','artifact']])}

## Dataset, chronology and populations

Reused `avm.clean_properties`, chronological splitting, `experiments.inputs`, deterministic feature engineering with valuation year 2026, and the local official district snapshot. Original records: {manifest['cleaning']['raw']:,}; cleaned single-family transactions: {manifest['cleaning']['deduplicated_transactions']:,}. Training is December 2025-March 2026; validation is April 2026; test is May 2026. All saved pipelines retain their original training-fitted preprocessing; no `fit` or `fit_transform` is called here. Target, date, asking prices and DOM are absent from model inputs.

All-cleaned counts: {manifest['chronology']['sizes_before_outlier_filter']}. Bounded counts: {manifest['bounds']['rows_after_outlier_filter']}. Bounded means actual price between ${manifest['bounds']['train_price_min']:,.0f} and ${manifest['bounds']['train_price_max']:,.0f}, inclusive, from the original training 0.5th/99.5th percentiles. The primary Week 8 reliability population is **all cleaned**; bounded results reproduce earlier comparisons. No extra quality filter, prediction clipping, or silent tail exclusion is applied. Both populations use identical frozen predictions. Target-based eligibility cannot be applied to a home whose sale price is unknown.

## Metric definitions and invalid-value handling

With residual e = prediction - actual: MAE = mean(|e|); RMSE = sqrt(mean(e²)); R2 = 1 - sum(e²)/sum((actual-mean(actual))²); MAPE = mean(100|e|/actual); MdAPE = median(100|e|/actual). Bias = mean(e), positive for overprediction. Percentage metrics use only finite positive actuals. Nonfinite actuals are counted and excluded from metrics; finite zero/negative actuals remain in dollar metrics but are counted as excluded from percentages. Nonfinite predictions fail evaluation instead of disappearing. The actual cleaned snapshot has no exclusions under this metric policy.

R2 is undefined for constant actuals or fewer than two observations. Price-band R2 additionally requires at least 50 observations; counts and an `R2_reason` accompany every row. This is a reporting guard, not a claim of statistical precision. Even with adequate counts, restricted price ranges make subgroup R2 unstable and often negative. No confidence intervals or significance claims are inferred from these single-month metrics.

## Training-derived price categories

Examined the entire cleaned training price distribution, including tails, before creating categories. Exact quantiles and a full-range chart are saved. The boundaries use linear-interpolated training Q25, Q50 and Q90, unchanged for April, May and both populations:

| Category | Actual-price interval | Approximate training share |
| --- | --- | --- |
| Lower-priced | below ${cuts[0]:,.2f} | 25% |
| Mid-range | ${cuts[0]:,.2f} to below ${cuts[1]:,.2f} | 25% |
| Upper-mid-range | ${cuts[1]:,.2f} to below ${cuts[2]:,.2f} | 40% |
| Luxury | at least ${cuts[2]:,.2f} | 10% |

Intervals are lower inclusive and upper exclusive; outer intervals are unbounded. Shares can differ because of ties. Luxury is a relative top-decile label for this training sample, not an external market classification. Bands describe retrospective errors by actual prices; they are not model inputs or prediction-time routing rules.

## Validation comparison and assessment before May

Individual metric leaders are reported separately. The balanced choice uses equal mean ranks for R2 (higher), MAE and MdAPE (lower) on all-cleaned April; model ID breaks exact rank ties. This transparent heuristic balances explained variation, average dollar error and typical relative error. RMSE and MAPE are still reported; RMSE is not ranked separately because it is monotonic with R2 on identical rows. Ranking tiny differences does not make them significant. `selection_before_may.json` was written before any May predictions; its hash is verified afterward. No May result changes this assessment.

'''
    for population in ('all_cleaned','bounded'):
        leaders = selection['leaders'][population]
        text += f"### April validation: {population}\n\n"
        for metric in ('R2','MAE_USD','MdAPE_pct'):
            text += f"- Best {metric}: {', '.join(leaders[metric])}.\n"
        text += f"- Balanced rank choice: {leaders['balanced']}.\n\n"
        table = summary.loc[(summary.split=='validation')&(summary.population==population)&(summary.price_band=='All')]
        text += markdown(table[['model_id','n','R2','MAE_USD','RMSE_USD','MAPE_pct','MdAPE_pct','bias_USD']])+'\n\n'
        unique_leaders = set(sum([leaders[k] for k in ('R2','MAE_USD','MdAPE_pct')],[]))
        text += ('Metric winners disagree; no universally best model is claimed.\n\n' if len(unique_leaders)>1 else
                 'These three metric leaders agree on this population, but that does not establish universal superiority.\n\n')
    text += '## May test: descriptive comparison of frozen models\n\nMay is reported separately, including for models not previously scored on May. It is not used to choose a feature set, hyperparameter or winner.\n\n'
    for population in ('all_cleaned','bounded'):
        table = summary.loc[(summary.split=='test')&(summary.population==population)&(summary.price_band=='All')]
        text += f'### May test: {population}\n\n'+markdown(table[['model_id','n','R2','MAE_USD','RMSE_USD','MAPE_pct','MdAPE_pct','bias_USD']])+'\n\n'
    text += f'## Price-band findings for the April balanced choice\n\nModel: **{selected}**. Every model has the same full band breakdown in `metrics_summary.csv`; the tables below focus on the prespecified April choice.\n\n'
    for split in ('validation','test'):
        for population in ('all_cleaned','bounded'):
            table = summary.loc[(summary.model_id==selected)&(summary.split==split)&
                                (summary.population==population)&(summary.price_band!='All')]
            text += f'### {split}: {population}\n\n'+markdown(table[['price_band','n','MAE_USD','RMSE_USD','MAPE_pct','MdAPE_pct','R2','R2_reason','bias_USD']])+'\n\n'
            high = table.loc[table.MdAPE_pct.idxmax()]
            indexed = table.set_index('price_band')
            text += f"Largest band MdAPE: **{high.price_band}**, {high.MdAPE_pct:.3f}%. Lower-priced mean bias is ${indexed.loc['Lower-priced','bias_USD']:,.0f}; luxury mean bias is ${indexed.loc['Luxury','bias_USD']:,.0f}. Positive indicates overprediction; negative indicates underprediction.\n\n"
    tails = pd.read_csv(out/'tail_sensitivity.csv')
    text += '## Tail sensitivity\n\n'
    for split in ('validation','test'):
        table = summary.loc[(summary.model_id==selected)&(summary.split==split)&(summary.price_band=='All')].set_index('population')
        tail = tails.loc[(tails.model_id==selected)&(tails.split==split)].iloc[0]
        text += f"For {split}, the chosen model has R2 {table.loc['bounded','R2']:.6f} on bounded sales versus {table.loc['all_cleaned','R2']:.6f} on all cleaned sales. The {int(tail.n_outside):,} outside-bound sales account for {tail.outside_SSE_share_pct:.3f}% of total squared error and {tail.outside_absolute_error_share_pct:.3f}% of absolute error. They remain in all-cleaned metrics.\n\n"
    text += '''## Figures and reproducibility

`outputs/week08/figures/` contains 200-DPI PNGs and PDF figures for all seven requested chart types, separately labeled for April/May and all-cleaned/bounded populations, plus the full training distribution. Actual/predicted and error-distribution plots retain the full range with labeled logarithmic/symmetric-log axes. Distribution bins have unequal dollar/percentage widths; heights are counts per bin. Price-band charts compare the April choice with the original baseline and the Week 6 reference. Model comparison charts show all saved pipelines.

Execute `notebooks/06_evaluation.ipynb` in the existing environment, or run `.\\.venv\\Scripts\\python.exe scripts/run_week8.py`, followed by `.\\.venv\\Scripts\\python.exe scripts/verify_week8.py`. The notebook runs the framework, displays executed results and figures, and verifies saved artifacts. `metrics_summary.csv` is under `outputs/week08/` so the historical `outputs/metrics_summary.csv` remains intact. The consolidated CSV includes every model, split, population, overall and price-band row. Prediction arrays and licensed aggregate results stay local. The verification manifest records input, source, output and historical hashes, model settings, versions and evaluation phase order.

## Limitations and recommendations

April has been used repeatedly for tuning and feature selection; the balanced ranking is exploratory, not an unbiased estimate of future performance. May has already been examined, and Week 8 does not restore its status as an untouched holdout. Training metrics include rows used for fitting and are diagnostics only. Models were originally trained on bounded targets; all-cleaned evaluation exposes tails the models were not trained to represent. Very small positive recorded prices inflate percentage errors; very large prices dominate squared errors. Review source-record validity under the data license, but retain them until an independently justified, documented quality rule is approved and tested without test-driven tuning.

District boundaries are retrospective 2024-25 polygons with a March 2026 service revision, and sold-snapshot attributes do not establish point-in-time availability. Districts encode location, not causal school quality. Error-band comparisons use actual prices and do not show geographic fairness, temporal stability or confidence intervals. No model is production-ready based on these two months.

Use later unseen months, rolling chronological backtests and geographic holdouts to confirm any model/feature choice. Prioritize the price bands and tail patterns identified above; examine both dollar and relative errors. Investigate robust or transformed-target approaches only in a future training/validation protocol, never by optimizing the May results in this report. Preserve the original raw data and all historical results.
'''
    (root/'docs/WEEK8_RESULTS.md').write_text(text,encoding='utf-8')
