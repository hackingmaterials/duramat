

import warnings

import pandas as pd
import numpy as np
from scipy import optimize
from sklearn import ensemble, metrics, model_selection
import matplotlib.pyplot as plt
import itertools
# import xgboost as xgb


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
    df['GHI>0'] = (df[meas_col] > 0).astype(int)

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
    df[ll_diff_label] = df[ll_diff_label] # * df[ll_diff_label]

    # |GHI - GHIcs|
    meas_model_abs_diff_label = 'abs(GHI-GHIcs)'
    calc_abs_diff(df, meas_col, model_col, meas_model_abs_diff_label, overwrite=overwrite)
    calc_window_stat(df, window, meas_model_abs_diff_label, meas_model_abs_diff_label, overwrite=overwrite)

    # GHImean - GHIcs mean
    avg_diff_label = 'avg(GHI)-avg(GHIcs)'
    df[avg_diff_label] = df[meas_col + ' mean'] - df[model_col + ' mean']

    max_diff_label = 'max(dGHI)-max(dGHIcs)'
    df[max_diff_label] = df[d_meas_label + ' max'] - df[d_model_label + ' max']

    # GHIstd - GHIcs std
    std_diff_label = 'std(GHI)-std(GHIcs) normed'
    # df[std_diff_label] = (df[meas_col + ' std'] / df[meas_col + ' mean']).replace([-np.inf, np.inf, np.nan], 0) -\
    #                      (df[model_col + ' std'] / df[model_col + ' mean']).replace([-np.inf, np.inf, np.nan], 0)
    df[std_diff_label] = \
        (df[d_meas_label + ' std'] / df[meas_col + ' mean']).replace([-np.inf, np.inf, np.nan], 0) -\
        (df[d_model_label + ' std'] / df[model_col + ' mean']).replace([-np.inf, np.inf, np.nan], 0)

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
    # df[label + ' mean'] = df[col].rolling(window, center=True, win_type='gaussian').mean(std=1).fillna(0)
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


class ProbabilityRandomForestClassifier(ensemble.RandomForestClassifier):

    def __init__(self, proba_cutoff=0.5, window=None,
                 n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=0, bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
                 verbose=0, warm_start=False, class_weight=None ):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes, # min_impurity_decrease=min_impurity_decrease,
                         min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
                         n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
                         class_weight=class_weight)
        self.proba_cutoff = proba_cutoff
        self.window = window

    def predict(self, X):
        labels = self.predict_proba(X)[:, 1] > self.proba_cutoff
        # labels = np.convolve(labels, np.ones(self.window) / self.window, mode='same')
        # labels = labels > self.proba_cutoff
        # slices = np.arange(len(labels))
        # slices = [slices[i: i + self.window] for i in range(len(labels) - self.window + 1)]
        # for i in range(self.window // 2):
        #     slices.insert(0, slices[0])
        #     slices.append(slices[-1])
        # labels = np.all(labels[slices], axis=1)
        return labels

# class ProbabilityXGBClassifier(xgb.XGBClassifier):
# 
#     def __init__(self, proba_cutoff=0.5, window=None,
#                  n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1,
#                  min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
#                  min_impurity_split=0, bootstrap=True, oob_score=False, n_jobs=1, random_state=None,
#                  verbose=0, warm_start=False, class_weight=None ):
#         super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
#                          min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
#                          min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
#                          max_leaf_nodes=max_leaf_nodes, # min_impurity_decrease=min_impurity_decrease,
#                          min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score,
#                          n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start,
#                          class_weight=class_weight)
#         self.proba_cutoff = proba_cutoff
#         self.window = window
# 
#     def predict(self, X):
#         labels = self.predict_proba(X)[:, 1] >= self.proba_cutoff
#         # labels = np.convolve(labels, np.ones(self.window) / self.window, mode='same')
#         # labels = labels > self.proba_cutoff
#         # slices = np.arange(len(labels))
#         # slices = [slices[i: i + self.window] for i in range(len(labels) - self.window + 1)]
#         # for i in range(self.window // 2):
#         #     slices.insert(0, slices[0])
#         #     slices.append(slices[-1])
#         # labels = np.all(labels[slices], axis=1)
#         return labels


def f1_tnr(ytrue, yhat):
    f1 = metrics.f1_score(ytrue, yhat)
    cm = metrics.confusion_matrix(ytrue, yhat)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    tnr = tn / (tn + fp)
    return f1 * tnr


def f1_tnr_tnr(ytrue, yhat):
    f1 = metrics.f1_score(ytrue, yhat)
    cm = metrics.confusion_matrix(ytrue, yhat)
    tn = cm[0, 0]
    fp = cm[0, 1]
    fn = cm[1, 0]
    tp = cm[1, 1]
    tnr = tn / (tn + fp)
    return f1 * tnr * tnr


def plot_confusion_matrix(cm, classes=['cloudy', 'clear'],
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Taken from sklearn docs (http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 4)

    fig, ax = plt.subplots(figsize=(6, 6))
    p = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(title)
    cb = fig.colorbar(p, fraction=0.046, pad=0.04)
    cb.ax.set_yticklabels(cb.ax.get_yticklabels(), fontsize=22)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=22)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=22)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], fontsize=22,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label', fontsize=22)
    ax.set_xlabel('Predicted label', fontsize=22)
    fig.tight_layout()


def plot_ml_vs_nsrdb(ground_obj, yhat, mask, tsplit, t0, t1,
                     yhat_label='ml_status', plot_label='ML clear',
                     fig_kwargs={}, fname='tmp.png'):
    df = ground_obj.df[~tsplit]
    df[yhat_label] = yhat

    df = df[(df.index >= t0) & (df.index < t1)]
    nplots = len(pd.unique(df.index.date))
    ncol = min(3, nplots)
    nrow = max(1, (nplots // ncol))
    try:
        figsize = fig_kwargs.pop('figsize')
    except KeyError:
        figsize = (ncol * 6, nrow * 6)

    fig, axes = plt.subplots(figsize=figsize, ncols=ncol, nrows=nrow, sharey=True)
    if ncol == 1:
        axes = np.asarray([axes])

    for i, (date, group) in enumerate(df.groupby(df.index.date)):
        group = group[(group.index.hour >= 4) & (group.index.hour <= 20)]
        ax = axes.ravel()[i]
        ax.set_title(date)
        ax.set_xticks([i for i in group.index.time if i.minute % 60 == 0][::2])
        ax.set_xticklabels([i for i in group.index.time if i.minute % 60 == 0][::2])
        if i % ncol == 0:
            ax.set_ylabel('GHI / Wm$^{-2}$')
        a = ax.plot(group.index.time, group['GHI'], label='GHI', c='k')
        b = ax.plot(group.index.time, group['Clearsky GHI pvlib'], label='GHIcs', c='k', alpha=.5)
        c = ax.plot(group.index.time, group['GHI nsrdb'], label='GHInsrdb', c='k', alpha=.5, linestyle='--')

        d = ax.scatter(group[group[yhat_label]].index.time, group[group[yhat_label]]['GHI'], label=plot_label)

        e = ax.scatter(group[group['sky_status'] & mask].index.time, group[group['sky_status'] & mask]['GHI'], label='NSRDB clear',
                   marker='o', s=250, facecolor='none', edgecolor='C1', linewidth=2)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='best', prop={'size': 14}, shadow=False, framealpha=1)
    # fig.text(.04, .5, 'GHI / Wm$^{-2}$', va='center', rotation='vertical')
    fig.tight_layout()
    fig.autofmt_xdate()
    fig.savefig(fname, dpi=300)




def plot_proba(ground_obj, probas, mask, tsplit, t0, t1):
    fig, ax = plt.subplots(figsize=(16, 6))

    df = ground_obj.df[~tsplit]
    df['proba'] = probas

    df = df[(df.index >= t0) & (df.index < t1)]

    ax.plot(df.index, df['GHI'], label='GHI', c='k')
    ax.plot(df.index, df['Clearsky GHI pvlib'], label='GHIcs', c='k', alpha=.5)

    cb = ax.scatter(df.index, df['GHI'], c=df['proba'], label='P$_{clear}$')

    ax.set_title('Probability of clear skies')
    fig.colorbar(cb)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('GHI / $\mathrm{Wm}^{-2}$')
    fig.tight_layout()
    fig.autofmt_xdate()


def my_kf_cv(clf, X_tr, y_tr, n_splits=3, max_depth_range=range(6, 21),
             proba_range=np.arange(.5, 1, .025), score_func=f1_tnr):
    results = []
    kfcv = model_selection.KFold(n_splits=n_splits)
    for depth in max_depth_range:
        clf.set_params(max_depth=depth)
        local_scores = {'probas': [],
                        'scores': []}
        for train, test in kfcv.split(X_tr):
            X_tr_cv, X_te_cv = X_tr[train], X_tr[test]
            y_tr_cv, y_te_cv = y_tr[train], y_tr[test]
            clf.fit(X_tr_cv, y_tr_cv)
            for cut in proba_range:
                cut = np.round(cut, 4)
                yhat = clf.predict_proba(X_te_cv)[:, 1] > cut
                # score = metrics.f1_score(y_te_cv, yhat)
                score = score_func(y_te_cv, yhat)
                local_scores['scores'].append(score)
                local_scores['probas'].append(cut)
        df = pd.DataFrame(local_scores)
        for proba, group in df.groupby('probas'):
            results.append({'max_depth': depth, 'proba_cutoff': proba,
                            'mean_scores': group['scores'].mean(), 'std_scores': group['scores'].std()})

    results_df = pd.DataFrame(results)
    return results_df[results_df['mean_scores'] == results_df['mean_scores'].max()].to_dict(orient='records')[0]

def plot_marked_masks(ground_obj, t0, t1, fname='tmp.png'):
    # fig, ax = plt.subplots(figsize=(20, 10), dpi=300)

    df = ground_obj.df
    df = df[(df.index >= t0) & (df.index < t1)]

    # ax.plot(df.index, df['GHI'], label='GHI', c='k')
    # ax.plot(df.index, df['Clearsky GHI pvlib'], label='GHIcs', c='k', alpha=.5)
    # ax.plot(df.index, df['GHI nsrdb'], label='GHInsrdb', c='k', alpha=.5, linestyle='--')

    # ax.scatter(df[df['sky_status'].astype(bool)].index, df[df['sky_status'].astype(bool)]['GHI'], label='NSRDB clear',
    #            marker='o', s=250, facecolor='none', edgecolor='C1', linewidth=2)

    use_masks = [i for i in ground_obj.masks_ if i not in ('empty_days', 'day_time')]
    # for m, mark in zip(ground_obj.masks_, ['o', 'x', '+', '^', '.']):
    #     ax.scatter(df[~df[m] & (df['GHI'] > 0)].index, df[~df[m] & (df['GHI'] > 0)]['GHI'], label=m, marker=mark, s=250, zorder=10)

    # df = ground_obj.df[~tsplit]
    # df[yhat_label] = yhat

    df = df[(df.index >= t0) & (df.index < t1)]
    nplots = len(pd.unique(df.index.date))
    ncol = min(3, nplots)
    nrow = max(1, (nplots // ncol))
    fig, axes = plt.subplots(figsize=(ncol * 6, (nrow) * 6), ncols=ncol, nrows=nrow, sharey=True)

    for i, (date, group) in enumerate(df.groupby(df.index.date)):
        group = group[(group.index.hour >= 4) & (group.index.hour <= 20)]
        ax = axes.ravel()[i]
        ax.set_title(date)
        ax.set_xticks([i for i in group.index.time if i.minute % 60 == 0][::2])
        ax.set_xticklabels([i for i in group.index.time if i.minute % 60 == 0][::2])
        a = ax.plot(group.index.time, group['GHI'], label='GHI', c='k')
        b = ax.plot(group.index.time, group['Clearsky GHI pvlib'], label='GHIcs', c='k', alpha=.5)
        c = ax.plot(group.index.time, group['GHI nsrdb'], label='GHInsrdb', c='k', alpha=.5, linestyle='--')
        e = ax.scatter(group[group['sky_status'].astype(bool)].index.time, group[group['sky_status'].astype(bool)]['GHI'],
                       label='NSRDB clear',
                       marker='o', s=250, facecolor='none', edgecolor='C1', linewidth=2)

        for m, mark, color in zip(use_masks, ['o', 'x', '+', '^', '.'], ['C2', 'C3', 'C4', 'C5', 'C6']):
            ax.scatter(group[~group[m] & (group['GHI'] > 0)].index.time,
                       group[~group[m] & (group['GHI'] > 0)]['GHI'], label=m, marker=mark, s=250, zorder=10, c=color)

        # if ax is axes[-1]:
        #     ax.legend()

        # d = ax.scatter(group[group[yhat_label]].index.time, group[group[yhat_label]]['GHI'], label=plot_label)


    handles, labels = ax.get_legend_handles_labels()
    # fig.subplots_adjust(right=.2)
    # fig.legend(handles, labels, loc='best', prop={'size': 14}, shadow=False, framealpha=1, bbox_to_anchor=(1.05, .9))
    fig.legend(handles, labels, loc='upper right', prop={'size': 14}, shadow=False, framealpha=1)
    fig.tight_layout()
    fig.autofmt_xdate()
    fig.savefig(fname, dpi=300)


    # ax.scatter(df[~df['sky_status'] & ~mask].index, df[~df['sky_status'] & ~mask]['GHI'], label='NSRDB cloudy (unmasked)',
    #            marker='x', s=250, c='C3', linewidth=2)


    # ax.legend()
    # ax.set_title('Ground and NSRDB Detection')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('GHI / $\mathrm{Wm}^{-2}$')
    # fig.tight_layout()
    # fig.autofmt_xdate()
    # fig.savefig(fname, dpi=300)

if __name__ == '__main__':
    main()
