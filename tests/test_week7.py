"""Meaningful guards on grouped shuffling and controlled removal definitions."""
import unittest
import numpy as np
import pandas as pd
from src import avm
from src.features import ENGINEERED, DISTRICTS
from src.week7 import feature_groups, permute_columns, subgroup_table, permutation, summarize_permutation


class Week7Tests(unittest.TestCase):
    def test_permutation_identifies_used_predictor_reproducibly(self):
        class KnownModel:
            def predict(self, frame):
                return frame['signal'].to_numpy()

        x = pd.DataFrame({'signal': np.arange(1., 21.), 'unused': np.arange(21., 41.)})
        groups = {'signal': ['signal'], 'unused': ['unused']}
        first = permutation(KnownModel(), x, x.signal, groups, repeats=3)
        second = permutation(KnownModel(), x, x.signal, groups, repeats=3)
        pd.testing.assert_frame_equal(first, second)
        summary = summarize_permutation(first)
        self.assertGreater(summary.loc['signal', 'R2_drop'], 0)
        self.assertEqual(summary.loc['unused', 'R2_drop'], 0)
        self.assertEqual(summary.loc['signal', 'best_rank'], 1)

    def test_joint_shuffle_preserves_pairs_and_index(self):
        x = pd.DataFrame({'a':[1,2,3], 'b':[10,20,30], 'untouched':[7,8,9]}, index=[5,5,9])
        changed = permute_columns(x, ['a','b'], np.array([2,0,1]))
        self.assertEqual(changed.a.tolist(), [3,1,2])
        self.assertTrue((changed.b == changed.a*10).all())
        pd.testing.assert_series_equal(x.untouched, changed.untouched)
        self.assertEqual(x.a.tolist(), [1,2,3])

    def test_ablation_information_definitions(self):
        num = avm.NUMERIC_CANDIDATES+ENGINEERED
        cat = avm.CATEGORICAL_CANDIDATES+DISTRICTS
        groups = feature_groups(num,cat)
        self.assertEqual(set(num+cat)-set(groups['all_engineered']),set(avm.NUMERIC_CANDIDATES+avm.CATEGORICAL_CANDIDATES))
        self.assertTrue(set(DISTRICTS+ENGINEERED[5:]).issubset(groups['all_geography']))
        self.assertFalse(set(num+cat)&avm.FORBIDDEN_FEATURES)

    def test_small_groups_suppressed_and_bias_sign(self):
        errors = pd.DataFrame({'district':['a']*50+['b'], 'actual':[100.]*51, 'predicted':[110.]*51})
        table = subgroup_table(errors,'district')
        self.assertEqual(table.group.tolist(), ['a'])
        self.assertEqual(table.bias_USD.iloc[0],10)
        self.assertEqual(table.MdAPE_pct.iloc[0],10)
