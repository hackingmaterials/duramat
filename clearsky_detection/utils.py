

import warnings

import pandas as pd
import numpy as np
from scipy import optimize


def main():
    pass


def calc_all_window_metrics(df, window, meas_col, model_col, overwrite=True):
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
    # dx = df.index.freq.nanos / 1.0e9 / 60.0e0
    dx = np.diff(df.index.values).min().astype(int) / 1.0e9 / 60.0e0

    calc_window_stat(df, window, meas_col, meas_col, overwrite=True)
    calc_window_stat(df, window, model_col, model_col, overwrite=True)
    df['GHI==0'] = df[meas_col] == 0

    # dGHI/dt
    d_meas_label = 'dGHI'
    # label = meas_col + ' gradient'
    df[d_meas_label] = df[meas_col].diff().fillna(method='bfill') / dx
    calc_window_stat(df, window, d_meas_label, d_meas_label, overwrite=True)

    # d2GHI/dt2
    # dd_meas_label = 'd2GHI'
    # label2 = label + ' second'
    # df[dd_meas_label] = df[d_meas_label].diff().fillna(method='bfill') / dx
    # df[dd_meas_label] = np.gradient(df[d_meas_label], dx)

    # dGHIcs/dt
    d_model_label = 'dGHIcs'
    df[d_model_label] = df[model_col].diff().fillna(method='bfill') / dx
    calc_window_stat(df, window, d_model_label, d_model_label, overwrite=True)

    # dGHIcs2/dt2
    # dd_model_label = 'd2GHIcs'
    # df[dd_model_label] = df[d_model_label].diff().fillna(method='bfill') / dx
    # df[dd_model_label] = np.gradient(df[d_model_label], dx)

    # GHI line length
    label_ll_1 = 'GHILL'
    calc_line_length(df, meas_col, window, dx, label=label_ll_1, overwrite=overwrite)

    # GHIcs line length
    label_ll_2 = 'GHIcsLL'
    calc_line_length(df, model_col, window, dx, label=label_ll_2, overwrite=overwrite)

    # (GHI LL / GHIcs LL)
    # ll_ratio_label = 'GHILL/GHIcsLL'
    # calc_ratio(df, label_ll_1, label_ll_2, ll_ratio_label, overwrite=overwrite)
    # df[ll_ratio_label] = df[ll_ratio_label].replace([-np.inf, np.inf, np.nan], -1)
    # df.loc[:, ll_ratio_label] = df['scale'] * df[ll_ratio_label]

    # GHI / GHIcs
    meas_model_ratio_label = 'GHI/GHIcs'
    calc_ratio(df, meas_col, model_col, meas_model_ratio_label, overwrite=overwrite)
    df.loc[(df[meas_col] == 0) & (df[model_col] == 0), meas_model_ratio_label] = 0
    df[meas_model_ratio_label] = df[meas_model_ratio_label].replace([-np.inf, np.inf, np.nan], -1)
    calc_window_stat(df, window, meas_model_ratio_label, meas_model_ratio_label, overwrite=overwrite)

    # GHI - GHIcs
    meas_model_diff_label = 'GHI-GHIcs'
    calc_diff(df, meas_col, model_col, meas_model_diff_label, overwrite=overwrite)
    calc_window_stat(df, window, meas_model_diff_label, meas_model_diff_label, overwrite=overwrite)

    # # dGHI - dGHIc s
    # dghi_diff_label = 'abs(dGHI-dGHIcs)'
    # calc_abs_diff(df, d_meas_label, d_model_label, dghi_diff_label, overwrite=overwrite)
    # calc_window_stat(df, window, dghi_diff_label, dghi_diff_label, overwrite=overwrite)

    # dGHI - dGHIc s
    dghi_diff_label = 'dGHI-dGHIcs'
    calc_diff(df, d_meas_label, d_model_label, dghi_diff_label, overwrite=overwrite)
    calc_window_stat(df, window, dghi_diff_label, dghi_diff_label, overwrite=overwrite)

    # ddGHI - ddGHIcs
    # d2ghi_diff_label = 'd2GHI-d2GHIcs'
    # calc_diff(df, dd_meas_label, dd_model_label, d2ghi_diff_label, overwrite=overwrite)
    # calc_window_stat(df, window, d2ghi_diff_label, d2ghi_diff_label, overwrite=overwrite)

    # (GHI LL - GHIcs LL)
    ll_diff_label = 'GHILL-GHIcsLL'
    calc_diff(df, label_ll_1, label_ll_2, ll_diff_label, overwrite=overwrite)
    df[ll_diff_label] = df[ll_diff_label]

    # |GHI - GHIcs|
    meas_model_abs_diff_label = 'abs(GHI-GHIcs)'
    calc_abs_diff(df, meas_col, model_col, meas_model_abs_diff_label, overwrite=overwrite)
    calc_window_stat(df, window, meas_model_abs_diff_label, meas_model_abs_diff_label, overwrite=overwrite)

    # GHImean - GHIcs mean
    avg_diff_label = 'avg(GHI)-avg(GHIcs)'
    df[avg_diff_label] = df[meas_col + ' mean'] - df[model_col + ' mean']

    # GHIstd - GHIcs std
    std_diff_label = 'std(GHI)-std(GHIcs) normed'
    df[std_diff_label] = (df[meas_col + ' std'] / df[meas_col + ' mean']).replace([-np.inf, np.inf, np.nan], 0) -\
                         (df[model_col + ' std'] / df[model_col + ' mean']).replace([-np.inf, np.inf, np.nan], 0)

    df['max(GHI)-max(GHIcs)'] = df[model_col + ' max'] - df[meas_col + ' max']

    #dGHImean - dGHIcs mean
    mean_d_diff_label = 'avg(dGHI)-avg(dGHIcs)'
    df[mean_d_diff_label] = df[d_meas_label + ' mean'] - df[d_model_label + ' mean']

    #dGHIstd - dGHIcs std
    std_diff_label = 'std(dGHI)-std(dGHIcs)'
    df[std_diff_label] = df[d_meas_label + ' std'] - df[d_model_label + ' std']

    calc_max_diff_changes(df, window, meas_col, model_col)


def calc_max_diff_changes(df, window, meas, model, label='max(abs(diff(GHI)-diff(GHIcs)))'):
    tmp1 = df[meas] - df[model]
    vals = tmp1.rolling(window, center=True).apply(lambda x: np.max(np.abs(np.diff(x))))
    df[label] = vals.fillna(0)


def calc_diff(df, meas, model, label='diff', overwrite=False):
    """

    Parameters
    ----------
    df: pd.DataFrame
        Must have meas and model columns.
    meas: str
        Column name for measured GHI values.
    model: str
        Column name for modeled GHI values.
    label: str
        Label for new difference column
    overwrite: bool
        Permission to overwite exists 'label' column (if it exists)
    """
    if label in df.keys() and not overwrite:
        raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
    diff = df[meas] - df[model]
    df[label] = diff


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
    # df[label + ' mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=.5).fillna(0)
    # df[label + ' gauss1 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=1).fillna(0)
    # df[label + ' gauss10 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=10).fillna(0)
    # df[label + ' gauss.1 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=.1).fillna(0)
    # df[label + ' gauss.5 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=.5).fillna(0)
    # df[label + ' gauss.01 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=.01).fillna(0)
    # df[label + ' gauss10 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=10).fillna(0)
    # df[label + ' gauss.1 mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=.1).fillna(0)
    df[label + ' mean'] = df[col].rolling(window, center=True).mean().fillna(0)
    df[label + ' std'] = df[col].rolling(window, center=True).std().fillna(0)
    df[label + ' max'] = df[col].rolling(window, center=True).max().fillna(0)
    df[label + ' min'] = df[col].rolling(window, center=True).min().fillna(0)
    # df[label + ' mad'] = df[col].rolling(window, center=True).apply(lambda x: np.median(np.abs(x - np.median(x)))).fillna(0)

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
    ratio = (df[num] / df[denom])# .replace([-np.inf, np.inf, np.nan], 1)  # replace with 1 for 0/0 case
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
    ratio = (df[num] / df[denom])# .replace([-np.inf, np.inf, np.nan], 1)
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
                   n_iter=20, tol=1.e-8, ml_label='sky_status iter', proba_cutoff=None):
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
    running_alpha = 1
    converged = False
    for it in range(n_iter):
        calc_all_window_metrics(day, window, meas_col=meas_col, model_col=model_col, overwrite=overwrite)
        if type(proba_cutoff) == float:
            try:
                X = day[feature_cols].values
                y_pred_proba = clf.predict_proba(X)[:, 1]
                y_pred = y_pred_proba >= proba_cutoff
            except ValueError:
                X = day[feature_cols]
                y_pred = clf.predict_proba(X)[:, 1] >= proba_cutoff
                # y_pred = clf.predict(X)
                # y_pred = np.round(y_pred).astype(bool)
        else:
            try:
                X = day[feature_cols].values
                y_pred = clf.predict(X)
            except ValueError:
                X = day[feature_cols]
                y_pred = clf.predict(X)
                y_pred = np.round(y_pred).astype(bool)
        clear_meas = day[y_pred.astype(bool)][meas_col]
        clear_model = day[y_pred.astype(bool)][model_col]
        alpha_last = alpha

        if len(clear_meas) == 0:
            alpha = 1
        else:
            def rmse(alp):
                sqr_err = (clear_meas - (alp * clear_model))**2
                return np.sqrt(np.mean(sqr_err))
            min_scalar = optimize.minimize_scalar(rmse)
            alpha = min_scalar.x
            running_alpha *= alpha

        day[ml_label] = y_pred
        if proba_cutoff is not None:
            day[ml_label + ' proba'] = y_pred_proba
        # if alpha > 1.15 or alpha < .85:
        if running_alpha > 1.15 or running_alpha < .85:
            warnings.warn('Large scaling value.  Day will not be further assessed or scaled.', RuntimeWarning)
            break
        day[model_col] = day[model_col] * alpha
        if np.abs(alpha - alpha_last) < tol:
            converged = True
            break
    if not converged:
        warnings.warn('Scaling did not converge.', RuntimeWarning)
    return day

if __name__ == '__main__':
    main()
