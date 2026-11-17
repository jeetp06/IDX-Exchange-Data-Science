"""Metric arithmetic, invalid-value policies, band boundaries and selection integrity."""
import unittest
import numpy as np
import pandas as pd
from src.evaluation import (regression_metrics, training_price_bands,
                            assign_price_bands, metric_rows, validation_assessment)


class EvaluationTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        m = regression_metrics([100, 200, 400], [110, 180, 440])
        self.assertAlmostEqual(m['MAE_USD'], 70/3)
        self.assertAlmostEqual(m['RMSE_USD'], np.sqrt(2100/3))
        self.assertAlmostEqual(m['R2'], 1-2100/(140000/3))
        self.assertAlmostEqual(m['MAPE_pct'], 10)
        self.assertAlmostEqual(m['MdAPE_pct'], 10)
        self.assertAlmostEqual(m['bias_USD'], 10)

    def test_zero_negative_and_invalid_actuals_are_audited(self):
        m = regression_metrics([0, -10, 100, np.nan, np.inf], [5, 0, 110, 1, 1])
        self.assertEqual((m['n_total'], m['n'], m['n_invalid_actual']), (5, 3, 2))
        self.assertEqual((m['n_percentage'], m['n_excluded_percentage']), (1, 2))
        self.assertAlmostEqual(m['MAE_USD'], 25/3)
        self.assertEqual(m['MAPE_pct'], 10)
        self.assertTrue(np.isnan(regression_metrics([0, -1], [1, 2])['MAPE_pct']))

    def test_bad_predictions_and_shapes_fail(self):
        for y, p in [([1,2], [1,np.inf]), ([1], [np.nan]), ([1], [1,2]), ([[1]], [[1]])]:
            with self.assertRaises(ValueError):
                regression_metrics(y, p)

    def test_r2_undefined_and_negative(self):
        self.assertEqual(regression_metrics([2,2], [2,2])['R2_reason'], 'constant_actuals')
        self.assertTrue(np.isnan(regression_metrics([2], [3])['R2']))
        self.assertTrue(np.isnan(regression_metrics([], [])['MAE_USD']))
        self.assertLess(regression_metrics([1,2,3], [10,10,10])['R2'], 0)
        self.assertTrue(np.isnan(regression_metrics([1,2], [1,2], minimum_r2_n=50)['R2']))

    def test_training_boundaries_and_all_tails_retained(self):
        specification = training_price_bands(np.arange(1., 101.))
        np.testing.assert_allclose(specification['cuts'], [25.75, 50.5, 90.1])
        assigned = assign_price_bands([.001,*specification['cuts'],1e12], specification)
        self.assertEqual(list(assigned), ['Lower-priced','Mid-range','Upper-mid-range','Luxury','Luxury'])
        rows = metric_rows([1,1000], [2,900], specification)
        self.assertEqual(sum(row['n'] for row in rows[1:]), rows[0]['n'])
        self.assertEqual(len(rows), 5)
        self.assertTrue(np.isnan(rows[2]['MAE_USD']))
        with self.assertRaises(ValueError):
            training_price_bands([1,1,1])

    def test_selection_excludes_test_and_exposes_tradeoffs(self):
        rows = []
        for population in ['all_cleaned', 'bounded']:
            for model, r2, mae, mdape in [('a', .8, 20, 8), ('b', .7, 10, 9)]:
                rows.append(dict(model_id=model, split='validation', population=population,
                                 price_band='All', R2=r2, MAE_USD=mae, MdAPE_pct=mdape))
        summary = pd.DataFrame(rows)
        _, first = validation_assessment(summary)
        test = pd.DataFrame([dict(model_id='b', split='test', population='all_cleaned',
                                 price_band='All', R2=1, MAE_USD=0, MdAPE_pct=0)])
        _, second = validation_assessment(pd.concat([summary,test], ignore_index=True))
        self.assertEqual(first, second)
        self.assertEqual(first['leaders']['all_cleaned']['R2'], ['a'])
        self.assertEqual(first['leaders']['all_cleaned']['MAE_USD'], ['b'])
        self.assertEqual(first['selected_model_id'], 'a')


if __name__ == '__main__':
    unittest.main()
