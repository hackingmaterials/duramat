"""Docstring."""

import warnings

import pandas as pd
import numpy as np
from scipy import optimize

# from sklearn.base import ClassifierMixin, TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import utils


class RandomForestClassifierPV(RandomForestClassifier):
    """Docstring."""

    def __init__(self, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False,
                 n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None,
                 by_day=True, model_col='Clearsky GHI pvlib', meas_col='GHI', scale_for_fit=True, feature_cols=None, target_col=None, window=3,
                 n_iter=20, pred_col='pred', tol=1.0e-8):
        """Docstring."""
        # RandomForestClassifier arguments
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                         min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start, class_weight=class_weight)
        # Project-specific arguments
        self.by_day = by_day
        self.model_col = model_col
        self.meas_col = meas_col
        self.scale_for_fit = scale_for_fit
        if feature_cols is None or target_col is None:
            raise ValueError('Must specify feature_cols and target_col.  These must be calculated by calc_all_window_metrics or else error will be thrown.')
        self.feature_cols = feature_cols
        self.target_cols = target_col
        self.window = window
        self.n_iter = n_iter
        self.pred_col = pred_col
        self.tol = tol

    def fit(self, X, y):
        """Docstring."""
        # X, y = check_X_y(X.values, y.values.flatten())
        if self.scale_for_fit:
            self._scale_model(X, y)

        self.X_ = X
        self.y_ = y

        utils.calc_all_window_metrics(X, self.window, meas_col=self.meas_col, model_col=self.model_col, overwrite=True)
        super().fit(X.values, y.values.flatten())

    def predict(self, X):
        """Docstring."""
        check_is_fitted(self, ['X_', 'y_'])

        X = check_array(X.values)

        final_dfs = []
        if self.by_day:
            groups = self.df.groupby(self.df.index.date)
            my_list = [day.copy() for (name, day) in groups]
        else:
            my_list = [self.df]
        if self.n_jobs != 1:
            pass
            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            # with cf.ProcessPoolExecutor() as executor:
            #     day_dfs = [executor.submit(utils.day_prediction, day, clf, feature_cols,
            #                                window, meas_col, model_col, overwrite=overwrite, n_iter=n_iter, tol=tol, ml_label=ml_label) for day in my_list]
            #     for future in cf.as_completed(day_dfs):
            #         final_dfs.append(future.result())
        else:
            final_dfs = [self._day_prediction(day) for day in my_list]
        if self.n_jobs != 1:
            X = pd.concat(final_dfs).sort_index()
        else:
            X = pd.concat(final_dfs)
        return X[self.pred_col]

    def _day_prediction(self, X):
        """Docstring."""
        alpha = 1
        converged = False
        for it in range(self.n_iter):
            utils.calc_all_window_metrics(X, self.window, meas_col=self.meas_col, model_col=self.model_col, overwrite=True)
            XX = X[self.feature_cols].values
            y_pred = super().predict(XX)
            clear_meas = X[y_pred][self.meas_col]
            clear_model = X[y_pred][self.model_col]
            alpha_last = alpha

            def rmse(alpha):
                sqr_err = (clear_meas - (alpha * clear_model))**2
                return np.sqrt(np.mean(sqr_err))

            min_scalar = optimize.minimize_scalar(rmse)
            alpha = min_scalar.x
            X[self.pred_col] = y_pred
            if alpha > 1.15 or alpha < .85:
                warnings.warn('Large scaling value.  Day will not be further assessed or scaled.', RuntimeWarning)
                break
            X[self.model_col] = X[self.model_col] * alpha
            if np.abs(alpha - alpha_last) < self.tol:
                converged = True
                break
        if not converged:
            warnings.warn('Scaling did not converge.', RuntimeWarning)
        return X

    def _scale_model(self, X, y=None):
        """Scale model values to measured values based on clear sky periods.

        Arguments
        ---------
        meas_col: str
            Column of measured GHI values.
        model_col: str
            Column of model GHI values.
        status_col: str
            Column of clear/cloudy classification (must be binary).
        """
        print(type(X), y.info())
        print(y.head())
        if y is not None:
            clear_meas = X[y][self.meas_col]
            clear_model = X[y][self.model_col]
        else:
            clear_meas = X[self.meas_col]
            clear_model = X[self.model_col]
        alpha = 1

        def rmse(alpha):
            sqr_err = (clear_meas - (alpha * clear_model))**2
            return np.sqrt(np.mean(sqr_err))

        alpha = optimize.minimize_scalar(rmse).x
        X.loc[:, self.model_col] = alpha * X[self.model_col]
