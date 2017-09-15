

import warnings

import pandas as pd
import numpy as np
from scipy import optimize


def calc_all_window_metrics(df, window, meas_col='GHI', model_col='Clearsky GHI',
                            abs_ideal_ratio_diff='abs_ideal_ratio_diff', overwrite=False):
    """Setup data frames for machine learning.

    Parameters
    ---------
    df: pd.DataFrame
        Time series data on GHI, GHIcs.
    window: int
        Size of window for statistics calculations.
    meas_col: str
        Numerator for ratio and abs_diff_ratio calculation.
    model_col: str
        Denominator for ratio and abs_diff_ratio calculation.
    abs_ideal_ratio_diff: str
        Label for absolute difference ratio columns.
    overwrite: bool
        Allow column to be overwritten.
    """
    if (abs_ideal_ratio_diff in df.keys()) and not overwrite:
        raise RuntimeError('A label already exists.  Set overwrite to True or pick new label name.')
    # dx = df.index.freq.nanos / 1.0e9 / 60.0e0
    dx = np.diff(df.index.values).min().astype(int) / 1.0e9 / 60.0e0

    # dGHI/dt, d2GHI/dt2
    label = meas_col + ' gradient'
    # df[label] = np.gradient(df[meas_col].values, dx)
    df[label] = df[meas_col].diff().fillna(method='bfill') / dx
    label2 = label + ' second'
    # df[label2] = np.gradient(df[label].values, dx)
    df[label2] = df[label].diff().fillna(method='bfill') / dx

    # dGHIcs/dt, dGHIcs2/dt2
    label = model_col + ' gradient'
    # df[label] = np.gradient(df[model_col].values, dx)
    df[label] = df[model_col].diff().fillna(method='bfill') / dx
    calc_window_stat(df, window, label, label, overwrite=overwrite)
    label2 = label + ' second'
    # df[label2] = np.gradient(df[label].values, dx)
    df[label2] = df[label].diff().fillna(method='bfill') / dx
    calc_window_stat(df, window, label2, label2, overwrite=overwrite)

    # |1 - GHI/GHIcs|
    calc_abs_ratio_diff(df, meas_col, model_col, abs_ideal_ratio_diff, overwrite=overwrite)
    df.loc[:, abs_ideal_ratio_diff] = df['irrad_scaler'] * df[abs_ideal_ratio_diff]
    calc_window_stat(df, window, abs_ideal_ratio_diff, abs_ideal_ratio_diff, overwrite=overwrite)

    # d|1 - GHI/GHIcs|/dt, d2|1 - GHI/GHIcs|/dt2
    label = abs_ideal_ratio_diff + ' gradient'
    # df[label] = np.gradient(df[abs_ideal_ratio_diff].values, dx)
    df[label] = df[abs_ideal_ratio_diff].diff().fillna(method='bfill') / dx
    calc_window_stat(df, window, label, label, overwrite=overwrite)
    label2 = label + ' second'
    # df[label2] = np.gradient(df[label].values, dx)
    df[label2] = df[label].diff().fillna(method='bfill') / dx
    calc_window_stat(df, window, label2, label2, overwrite=overwrite)

    # \1 - (dGHI / dt) / (dGHIcs / dt)|, \1 - (d2GHI / dt2) / (d2GHIcs / dt2)|
    calc_abs_ratio_diff(df, meas_col + ' gradient', model_col + ' gradient',
                        meas_col + ' ' + model_col + ' gradient ratio', overwrite=overwrite)

    df.loc[:, meas_col + ' ' + model_col + ' gradient ratio'] = \
        df['irrad_scaler'] * df[meas_col + ' ' + model_col + ' gradient ratio']

    # calc_ratio(df, meas_col + ' gradient', model_col + ' gradient',
    #            meas_col + ' ' + model_col + ' gradient ratio', overwrite=overwrite)
    calc_window_stat(df, window, meas_col + ' ' + model_col + ' gradient ratio',
                     meas_col + ' ' + model_col + ' gradient ratio', overwrite=overwrite)
    calc_abs_ratio_diff(df, meas_col + ' gradient second',
                        model_col + ' gradient second', meas_col + ' ' + model_col + ' gradient second ratio',
                        overwrite=overwrite)
    # calc_ratio(df, meas_col + ' gradient second', model_col + ' gradient second',
    #            meas_col + ' ' + model_col + ' gradient second ratio', overwrite=overwrite)

    # df.loc[:, meas_col + ' ' + model_col + ' gradient second ratio'] = \
    #     df['irrad_scaler'] * df[meas_col + ' ' + model_col + ' gradient second ratio']

    calc_window_stat(df, window, meas_col + ' ' + model_col + ' gradient second ratio',
                     meas_col + ' ' + model_col + ' gradient second ratio', overwrite=overwrite)

    # GHI line length
    label_ll_1 = meas_col + ' line length'
    calc_line_length(df, meas_col, window, dx, label=label_ll_1, overwrite=overwrite)

    # GHIcs line length
    label_ll_2 = model_col + ' line length'
    calc_line_length(df, model_col, window, dx, label=label_ll_2, overwrite=overwrite)

    # GHI LL/GHIcs LL
    # calc_ratio(df, label_ll_1, label_ll_2, abs_ideal_ratio_diff + ' line length', overwrite=overwrite)

    # GHI LL / GHIcs LL, d(GHI LL / GHIcs LL) / dt
    label = meas_col + ' ' + model_col + ' line length ratio'
    calc_abs_ratio_diff(df, label_ll_1, label_ll_2, label, overwrite=overwrite)
    # calc_ratio(df, label_ll_1, label_ll_2, label, overwrite=overwrite)

    df.loc[:, label] = df['irrad_scaler'] * df[label]

    label1 = label + ' gradient'
    # df[label1] = np.gradient(df[label].values, dx)
    df[label1] = df[label].diff().fillna(method='bfill') / dx

    df.loc[:, label1] = df['irrad_scaler'] * df[label1]

    label2 = label1 + ' second'
    # df[label2] = np.gradient(df[label1].values, dx)
    df[label2] = df[label1].diff().fillna(method='bfill') / dx

    df.loc[:, label2] = df['irrad_scaler'] * df[label2]


def calc_window_stat(df, window, col, label, overwrite=False):
    """Calculates window-based statistics on columns.

    Parameters
    ---------
    df: pd.Dataframe
        Variable col must be in columns.
    window: int
        Size of window for calculating metrics.  Defined as the number of timesteps to use in the window.
    col: str
        Which column to use.
    label: str
        Result column label.
    overwrite: bool
        Allow columns to be overwritten if they exist.
    """
    test_labels = [label + i for i in [' mean', ' std', ' max', ' min', ' range']]
    if any(i in df.keys() for i in test_labels) and not overwrite:
        raise RuntimeError('A label already exists.  Use new label name set overwrite to True.')
    df[label + ' mean'] = df[col].rolling(window, center=True).mean().fillna(0)
    df[label + ' std'] = df[col].rolling(window, center=True).std().fillna(0)
    df[label + ' max'] = df[col].rolling(window, center=True).max().fillna(0)
    df[label + ' min'] = df[col].rolling(window, center=True).min().fillna(0)


def calc_abs_diff(df, num, denom, label='abs_diff', overwrite=False):
    """Absolute difference of two columns.

    Parameters
    ---------
    df: pd.Dataframe
        Must contain num/denom columns.
    num: str
        Column for numerator.
    denom: str
        Column for denominator.
    label: str
        Name of resultant column.
    overwrite: bool
        Permission to overwite existing 'label' column.
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    absdiff = np.abs(df[num] - df[denom])  # replace with 1 for 0/0 case
    df[label] = absdiff


def calc_ratio(df, num, denom, label='ratio', overwrite=False):
    """Ratio of two columns of dataframe.

    Parameters
    ---------
    df: pd.Dataframe
        Must include num/denom in column names.
    num: str
        Column for numerator.
    denom: str
        Column for denominator.
    label: str
        Name of resultant column.
    overwrite: bool
        Permission to overwrite column if it exists.
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    ratio = (df[num] / df[denom]).replace([-np.inf, np.inf, np.nan], 0)  # replace with 1 for 0/0 case
    df[label] = ratio


def calc_abs_ratio_diff(df, num, denom, label='abs_diff_ratio', overwrite=False):
    """Absolute difference of the ratio of two series from 1.

    Parameters
    ---------
    df: pd.Dataframe
        Must contain num/denom column names.
    num: str
        Column for numerator.
    denom: str
        Column for denominator.
    label: str
        Name of resultant column.
    overwrite: bool
        Overwite column if it exists.
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    ratio = (df[num] / df[denom]).replace([-np.inf, np.inf, np.nan], 1)
    df[label] = np.abs(1 - ratio)


def calc_line_length(df, col, window, dx, label=None, overwrite=False):
    """Calculate the rolling line length of data series.

    Parameters
    ---------
    df: pd.Dataframe
    col: str
        Data column to calculate for.
    window: int
        Number of time periods to include in window.
    dx: float
        Delta of x values (time in this case).
    label: str
        Name of resultant column.  If None, will add 'line length' to col.
    overwrite: bool
        Overwrite label if it exists.
    """
    if label is None:
        label = col + ' line length'
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    df[label] = df[col].rolling(window, center=True).apply(lambda x: line_length(x, dx)).fillna(0)


def line_length(x, dx):
    """Calculate line length traveled by array x (with delta x = dx).

    Parameters
    ---------
    x: np.ndarray
        1 dimensional array of numeric values.
    dx: float
        Difference of x values.

    Returns
    -------
    length: float
        Distsance curve travels.
    """
    ydiffs = np.diff(x)
    xdiffs = np.asarray([dx for _ in range(len(ydiffs))])
    length = np.sum(np.sqrt(ydiffs**2 + xdiffs**2))
    return length


def day_prediction(day, clf, feature_cols, window, meas_col, model_col, overwrite=False,
                   n_iter=20, tol=1.e-8, ml_label='sky_status iter'):
    """Make predictions on given dataframe.

    Parameters
    ----------
    day: pd.DataFrame
        Dataframe of time period to be classified.
    clf: sklearn estimator
        Must have predict method.
    feature_cols: list-like
        Strings of columns (in day dataframe) to use as features in classifier.
    window: int
        Size of window for classification.
    meas_col: str
        Column name of measured GHI.
    model_col: str
        Column name of modeled GHI.
    overwrite: bool
        Permission to overwrite existing columns in day (generally should be True).
    n_iter: int
        Maximum allowed iterations for predict/curve fit cycle.
    tol: float
        Convergence tolerance for ending predict/cuve fit cycle.
    ml_label: str
        Name of column of predictions.
    Returns
    -------
    day: pd.DataFrame
        Feature columns and ml_label columns will be appended/modified as needed.
    """
    alpha = 1
    converged = False
    for it in range(n_iter):
        calc_all_window_metrics(day, window, meas_col=meas_col, model_col=model_col, overwrite=overwrite)
        X = day[feature_cols].values
        y_pred = clf.predict(X)
        clear_meas = day[y_pred][meas_col]
        clear_model = day[y_pred][model_col]
        alpha_last = alpha

        def rmse(alp):
            sqr_err = (clear_meas - (alp * clear_model))**2
            return np.sqrt(np.mean(sqr_err))

        min_scalar = optimize.minimize_scalar(rmse)
        alpha = min_scalar.x
        day[ml_label] = y_pred
        if alpha > 1.15 or alpha < .85:
            warnings.warn('Large scaling value.  Day will not be further assessed or scaled.', RuntimeWarning)
            break
        day[model_col] = day[model_col] * alpha
        if np.abs(alpha - alpha_last) < tol:
            converged = True
            break
    if not converged:
        warnings.warn('Scaling did not converge.', RuntimeWarning)
    return day

# def iter_predict(self, feature_cols, meas_col, model_col, clf, window, n_iter=20,
#                  tol=1.0e-8, ml_label='sky_status iter', smooth=False, overwrite=True):
#     """Predict clarity based using classifier that iteratively fits the model column
#     to the measured column based on clear points.
#
#     This function WILL overwrite columns that already exist in the data frame.
#
#     Arguments
#     ---------
#     feature_cols: list-like
#         Column names to use as features in ML model.
#     clf: sklearn estimator
#         Object with fit and predict methods.
#     meas_col: str
#         Column of measured data.
#     model_col: str
#         Column of model data.
#     n_iter, optional: int
#         Number of iterations for fitting model to measured column.
#     tol, optoinal: float
#         Criterion for convergence of modeled and measured clear points.
#     ml_label, optional: str
#         Label for predicted clear/cloudy points.
#     smooth, optional: bool
#         Smooth results.  Smoothing is aggressive as a point must be clear
#         in every window it appears in.
#     overwrite, optional: bool
#         Permission to overwrite columns if they exist.
#     """
#     alpha = 1
#     for it in range(n_iter):
#         print(it + 1, alpha)
#         self.calc_all_window_metrics(window, col1=meas_col, col2=model_col,
#                                      ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=overwrite)
#         X = self.df[feature_cols].values
#         y_pred = clf.predict(X)
#         clear_meas = self.df[y_pred][meas_col]
#         clear_model = self.df[y_pred][model_col]
#         alpha_last = alpha
#         def rmse(alpha):
#             return np.sqrt(np.mean((clear_meas - (alpha * clear_model))**2))
#         min_scalar = optimize.minimize_scalar(rmse)
#         alpha = min_scalar.x
#         if np.abs(alpha - alpha_last) < tol:
#             break
#         self.df[model_col] = self.df[model_col] * alpha
#     if it == n_iter - 1:
#         warnings.warn('Scaling did not converge.', RuntimeWarning)
#     self.df[ml_label] = y_pred
#     if smooth:
#         self.smooth_ml_label(window * 2, ml_label)
#     return self.df[ml_label]

# def iter_predict_daily(self, feature_cols, meas_col, model_col, clf, window, n_iter=20,
#                              tol=1.0e-8, ml_label='sky_status iter', smooth=False, overwrite=True):
#     """Predict clarity based using classifier that iteratively fits the model column
#     to the measured column based on clear points.
#
#     This method differs from iter_predict method because it predicts/scales on
#     individual days, not the entire data set.
#     This function WILL overwrite columns that already exist in the data frame.
#
#     Method should be refactored.  Calculating features and scaling model GHI is done across entire data set
#     even though we only need one day at a time.
#
#     Arguments
#     ---------
#     feature_cols: list-like
#         Column names to use as features in ML model.
#     clf: sklearn estimator
#         Object with fit and predict methods.
#     meas_col: str
#         Column of measured data.
#     model_col: str
#         Column of model data.
#     n_iter, optional: int
#         Number of iterations for fitting model to measured column.
#     tol, optoinal: float
#         Criterion for convergence of modeled and measured clear points.
#     ml_label, optional: str
#         Label for predicted clear/cloudy points.
#     smooth, optional: bool
#         Smooth results.  Smoothing is aggressive as a point must be clear
#         in every window it appears in.
#     overwrite, optional: bool
#         Permission to overwrite columns if they exist.
#     """
#     # dx = self.df.index.freq.nanos / 1.0e9 / 60.0e0
#     day_dfs = []
#     groups = self.df.groupby(self.df[[meas_col, model_col]].index.date)
#     my_list = [day.copy() for (name, day) in groups]
#     # for name, day in self.df.groupby(self.df[[meas_col, model_col]].index.date):
#     for day in my_list:
#         alpha = 1
#         converged = False
#         for _ in range(n_iter):
#             utils.calc_all_window_metrics(day, window, meas_col=meas_col, model_col=model_col, overwrite=overwrite)
#             X = day[feature_cols].values
#             y_pred = clf.predict(X)
#             clear_meas = day[y_pred][meas_col]
#             clear_model = day[y_pred][model_col]
#             alpha_last = alpha
#             def rmse(alpha):
#                 sqr_err = (clear_meas - (alpha * clear_model))**2
#                 return np.sqrt(np.mean(sqr_err))
#             min_scalar = optimize.minimize_scalar(rmse)
#             alpha = min_scalar.x
#             day.loc[:, ml_label] = y_pred
#             if alpha > 1.25 or alpha < .75:
#                 warnings.warn('Large scaling value.  Day will not be further assessed or scaled.', RuntimeWarning)
#                 break
#             day.loc[:, model_col] = day[model_col] * alpha
#             if np.abs(alpha - alpha_last) < tol:
#                 converged = True
#                 break
#         if not converged:
#             warnings.warn('Scaling did not converge.', RuntimeWarning)
#         day_dfs.append(day)
#     indices = self.df.index
#     self.df = pd.concat(day_dfs)
#     self.df.index = indices
#     return self.df[ml_label]
