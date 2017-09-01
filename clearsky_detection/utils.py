

import warnings

import pandas as pd
import numpy as np
from scipy import optimize

def calc_all_window_metrics(df, window, meas_col='GHI', model_col='Clearsky GHI', abs_ideal_ratio_diff='abs_ideal_ratio_diff', overwrite=False):
    """Setup dataframes for machine learning.

    Arguments
    ---------
    df: pd.DataFrame
        Time series data on GHI, GHIcs.
    window: int
        Size of window for statistics calculations.
    meas_col: str
        Numerator for ratio and abs_diff_ratio calculation.
    model_col: str
        Denominator for ratio and abs_diff_ratio calculation.
    abs_ratio_diff_label: str
        Label for absolute difference ratio columns.
    overwrite, optional: bool
        Allow column to be overwritten.
    """
    if (abs_ideal_ratio_diff in df.keys()) and not overwrite:
        raise RuntimeError('A label already exists.  Set overwrite to True or pick new label name.')
    dx = df.index.freq.nanos / 1.0e9 / 60.0e0

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
    calc_abs_ratio_diff(df, meas_col + ' gradient', model_col + ' gradient', meas_col + ' ' + model_col + ' gradient ratio', overwrite=overwrite)
    calc_window_stat(df, window, meas_col + ' ' + model_col + ' gradient ratio', meas_col + ' ' + model_col + ' gradient ratio', overwrite=overwrite)
    calc_abs_ratio_diff(df, meas_col + ' gradient second', model_col + ' gradient second', meas_col + ' ' + model_col + ' gradient second ratio', overwrite=overwrite)
    calc_window_stat(df, window, meas_col + ' ' + model_col + ' gradient second ratio', meas_col + ' ' + model_col + ' gradient second ratio', overwrite=overwrite)

    # GHI line length
    label_ll_1 = meas_col + ' line length'
    calc_line_length(df, meas_col, window, dx, label=label_ll_1, overwrite=overwrite)

    # GHIcs line length
    label_ll_2 = model_col + ' line length'
    calc_line_length(df, model_col, window, dx, label=label_ll_2, overwrite=overwrite)

    # GHI LL/GHIcs LL
    calc_ratio(df, label_ll_1, label_ll_2, abs_ideal_ratio_diff + ' line length', overwrite=overwrite)

    # GHI LL / GHIcs LL, d(GHI LL / GHIcs LL) / dt
    label = meas_col + ' ' + model_col + ' line length ratio'
    calc_ratio(df, label_ll_1, label_ll_2, label, overwrite=overwrite)
    label1 = label + ' gradient'
    # df[label1] = np.gradient(df[label].values, dx)
    df[label1] = df[label].diff().fillna(method='bfill') / dx
    label2 = label1 + ' second'
    # df[label2] = np.gradient(df[label1].values, dx)
    df[label2] = df[label1].diff().fillna(method='bfill') / dx


def calc_window_stat(df, window, col, label, overwrite=False):
    """Calculates window-based statistics on columns.

    Arguments
    ---------
    window: int
        Size of window for calculating metrics.  Defined as the number of timesteps to use in the window.
        For example, if data is 30 min frequency and window=3, the window will span 1 hour.
    col: str
        Which column to use.
    label: str
        Result column label.
    overwrite, optional: bool
        Allow columns to be overwritten if they exist.
    """
    test_labels = [label + i for i in [' mean', ' std', ' max', ' min', ' range']]
    if any(i in df.keys() for i in test_labels) and not overwrite:
        raise RuntimeError('A label already exists.  Use new label name set overwrite to True.')
    # df[label + ' mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=1.0e0).fillna(0)
    df[label + ' mean'] = df[col].rolling(window, center=True).mean().fillna(0)
    df[label + ' std']  = df[col].rolling(window, center=True).std().fillna(0)
    df[label + ' max']  = df[col].rolling(window, center=True).max().fillna(0)
    df[label + ' min']  = df[col].rolling(window, center=True).min().fillna(0)

def calc_abs_diff(df, num, denom, label='abs_diff', overwrite=False):
    """Absolute difference of two columns.

    Arguments
    ---------
    num: str
        Column for numerator.
    denom: str
        Column for denominator.
    label: str
        Name of resultant column.
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    absdiff = np.abs(df[num] - df[denom])  # replace with 1 for 0/0 case
    df[label] = absdiff

def calc_ratio(df, num, denom, label='ratio', overwrite=False):
    """Ratio of two columns of dataframe.

    Arguments
    ---------
    num: str
        Column for numerator.
    denom: str
        Column for denominator.
    label: str
        Name of resultant column.
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    ratio = (df[num] / df[denom]).replace([-np.inf, np.inf, np.nan], 0)  # replace with 1 for 0/0 case
    df[label] = ratio

def calc_abs_ratio_diff(df, num, denom, label='abs_diff_ratio', overwrite=False):
    """Absolute difference of the ratio of two series from 1.

    Arguments
    ---------
    num: str
        Column for numerator.
    denom: str
        Column for denominator.
    label: str
        Name of resultant column.
    overwrite, optional: bool
        Overwite column if it exists.
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    ratio = (df[num] / df[denom]).replace([-np.inf, np.inf, np.nan], 1)
    df[label] = np.abs(1 - ratio)

def calc_line_length(df, col, window, dx, label=None, overwrite=False):
    """Calculate the rolling line length of data series.

    Arguments
    ---------
    col: str
        Data column to calculate for.
    window: int
        Number of time periods to include in window.
    dx: float
        Delta of x values (time in this case).
    label, optional: str
        Name of resultant column.  If None, will add 'line length' to col.
    overwrite, optional: bool
        Overwrite label if it exists.
    """
    if label is None:
        label = col + ' line length'
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    df[label] = df[col].rolling(window, center=True).apply(lambda x: line_length(x, dx)).fillna(0)

def line_length(x, dx):
    """Calculate line length traveled by array x (with delta x = dx).

    Arguments
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
    xdiffs = np.asarray([dx for i in range(len(ydiffs))])
    length = np.sum(np.sqrt((ydiffs)**2 + (xdiffs)**2))
    return length

def day_prediction(day, clf, feature_cols, window, meas_col, model_col, overwrite=False, n_iter=20, tol=1.e-8, ml_label='sky_status iter'):
    alpha = 1
    converged = False
    for _ in range(n_iter):
        calc_all_window_metrics(day, window, meas_col=meas_col, model_col=model_col, overwrite=overwrite)
        X = day[feature_cols].values
        y_pred = clf.predict(X)
        clear_meas = day[y_pred][meas_col]
        clear_model = day[y_pred][model_col]
        alpha_last = alpha
        def rmse(alpha):
            sqr_err = (clear_meas - (alpha * clear_model))**2
            return np.sqrt(np.mean(sqr_err))
        min_scalar = optimize.minimize_scalar(rmse)
        alpha = min_scalar.x
        day[ml_label] = y_pred
        if alpha > 1.1 or alpha < .9:
            # warnings.warn('Large scaling value.  Day will not be further assessed or scaled.', RuntimeWarning)
            break
        day[model_col] = day[model_col] * alpha
        if np.abs(alpha - alpha_last) < tol:
            converged = True
            break
    # if not converged:
    #     pass
        # warnings.warn('Scaling did not converge.', RuntimeWarning)
    return day
