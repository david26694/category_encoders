import pandas as pd
import numpy as np

import unittest

import category_encoders as encoders


class TestNestedTargetEncoder(unittest.TestCase):
    """Tests for nested target encoder."""

    def setUp(self):
        """Create dataframe with categories and a target variable"""

        self.col = 'col_1'
        self.parent_col = 'parent_col_1'
        self.X = pd.DataFrame({
            self.col: ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd', 'd'],
            self.parent_col: ['e', 'e', 'e', 'e', 'e', 'f', 'f', 'f', 'f', 'f']
        })

        self.X_array = pd.DataFrame({
            self.col: ['a', 'a', 'b', 'b', 'b', 'c', 'c', 'd', 'd', 'd'],
            self.parent_col: ['e', 'e', 'e', 'e', 'e', 'f', 'f', 'f', 'f', 'f']
        }).values

        self.y = pd.Series([1, 2, 3, 1, 2, 4, 4, 5, 4, 4.5])

        self.parent_means = list(self.y.groupby(self.X[self.parent_col]).mean())
        self.parents = ['e', 'f']

    def test_parent_prior(self):
        """
        Simple case:
        There is no prior from the global to the group mean (m_prior = 0).
        As the m_parent is 1, the mean for group a is (as mean_group_e = 1.8):
        (1 + 2 + mean_group_e ) / 3 = (1 + 2 + 1.8) / 3 = 1.6
        The same works for b, c and d
        """
        expected_output = pd.DataFrame(dict(
            col_1=[1.6, 1.6, 1.95, 1.95, 1.95, 4.1, 4.1, 4.45, 4.45, 4.45],
            parent_col_1=self.X[self.parent_col]
        ))

        te = encoders.NestedTargetEncoder(
            cols=self.col,
            parent_dict=dict(col_1=self.parent_col),
            m_prior=0
        )
        pd.testing.assert_frame_equal(
            te.fit_transform(self.X, self.y),
            expected_output
        )

    def test_numpy_array(self):
        """
        Check that nested target encoder also works for numpy arrays
        """
        expected_output = pd.DataFrame(dict(
            col_1=[1.6, 1.6, 1.95, 1.95, 1.95, 4.1, 4.1, 4.45, 4.45, 4.45],
            parent_col_1=self.X[self.parent_col]
        )).values

        te = encoders.NestedTargetEncoder(
            cols=0,
            parent_dict={0: 1},
            m_prior=0
        )

        te.fit(self.X_array, self.y)
        output = te.transform(self.X_array).values

        np.testing.assert_almost_equal(
            output[:, 0],
            expected_output[:, 0]
        )

        np.testing.assert_equal(
            output[:, 1],
            expected_output[:, 1]
        )

    def test_no_parent(self):
        """
        When using no priors, the functionalities should be the same as for
        m estimator.
        """

        te = encoders.NestedTargetEncoder(
            cols=self.col,
            parent_dict=dict(col_1=self.parent_col),
            m_prior=0,
            m_parent=0
        )

        m_te = MEstimateEncoder(
            cols=self.col,
            m=0
        )
        pd.testing.assert_frame_equal(
            te.fit_transform(self.X, self.y),
            m_te.fit_transform(self.X, self.y)
        )

    def test_unknown_missing_imputation(self):
        """
        When new categories and unknown values are given, we expect the encoder
        to give the parent means (at least with default configuration).
        """

        # First two rows are new categories
        # Last two rows are missing values
        # Parents are e, f, e, f
        new_x = pd.DataFrame({
            self.col: ['x', 'y', np.NaN, np.NaN],
            self.parent_col: self.parents + self.parents
        })

        # We expect to get parent means
        expected_output_df = pd.DataFrame({
            self.col: self.parent_means + self.parent_means,
            self.parent_col: self.parents + self.parents
        })

        te = encoders.NestedTargetEncoder(
            cols=self.col,
            parent_dict=dict(col_1=self.parent_col),
            m_prior=0
        )

        te.fit(self.X, self.y)

        pd.testing.assert_frame_equal(
            te.transform(new_x),
            expected_output_df
        )

    def test_missing_na(self):
        """
        When new categories and unknown values are given, we expect the encoder
        to give the parent means. If we specify return_nan, we want it to
        return nan
        """

        # First two rows are new categories
        # Last two rows are missing values
        # Parents are e, f, e, f
        new_x = pd.DataFrame({
            self.col: ['x', 'y', np.nan, np.nan],
            self.parent_col: self.parents + self.parents
        })

        # In the transformer we specify unknown -> return nan
        # We expect to get:
        # - nan for the unknown
        # - parent means for the missing
        expected_output_df = pd.DataFrame({
            self.col: [np.nan, np.nan] + self.parent_means,
            self.parent_col: self.parents + self.parents
        })

        te = encoders.NestedTargetEncoder(
            cols=self.col,
            parent_dict=dict(col_1=self.parent_col),
            m_prior=0,
            handle_missing='value',
            handle_unknown='return_nan'
        )

        te.fit(self.X, self.y)

        pd.testing.assert_frame_equal(
            te.transform(new_x),
            expected_output_df
        )

    def test_all_missing(self):
        """
        If everything's missing or unknow , we expect by default to return
        global mean
        """
        new_x = pd.DataFrame({
            self.col: ['x', np.nan, 'x', np.nan],
            self.parent_col: ['z', 'z', np.nan, np.nan]
        })

        te = encoders.NestedTargetEncoder(
            cols=self.col,
            parent_dict=dict(col_1=self.parent_col),
            m_prior=0
        )

        te.fit(self.X, self.y)

        expected_output_df = pd.DataFrame({
            self.col: self.y.mean(),
            self.parent_col: ['z', 'z', np.nan, np.nan]
        })

        pd.testing.assert_frame_equal(
            te.transform(new_x),
            expected_output_df
        )
