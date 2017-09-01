

import os
import itertools

import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import metrics

def read_dir_nsrdb(dir_path, timezone, file_ext='csv'):
    '''Read directory of NSRDB files into a dataframe.

    Arguments
    ---------
    dir_path: str
        Path to directory of files.
    timezone: pytz.timezone
        Timezone for the dataframe indicies.
    file_ext, optional: str
        Filetype to specify for reading.

    Returns
    -------
    df: pd.DataFrame
        Contains all fields from files.
    '''
    if file_ext.lower() not in ('csv'):
        raise NotImplementedError('Only accept CSV files at this time.')
    files = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(file_ext)]
    df = pd.concat([pd.read_csv(f, header=2) for f in files])
    df.index = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(timezone)
    df = df.sort_index()
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='30min')).fillna(0)
    df = df[~df.index.duplicated(keep='first')]
    return df

def read_snl_rtc(file_w_path, timezone1, timezone2):
    '''Read SNL RTC data into file.

    Arguments
    ---------
    file_w_path: str
        Path to file (absolute).
    timezone1: pytz.timezone or str
        Timezone for localization.
    timezone2: pytz.timezone or str
        Timezone to which indices will be converted.

    Returns
    -------
    df: pd.DataFrame
        Contains all fields from files.
    '''
    df = pd.read_csv(file_w_path, parse_dates=['Date-Time'], index_col=['Date-Time'])
    df.index = df.index.tz_localize(timezone1)
    df.index = df.index.tz_convert(timezone2)
    df = df.sort_index()
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')).fillna(0)
    df = df[~df.index.duplicated(keep='first')]
    return df

def read_srrl_dir(dir_path, timezone, file_ext='txt'):
    '''Read directory of SRRL files into a dataframe.

    Arguments
    ---------
    dir_path: str
        Path to directory of files.
    timezone: pytz.timezone
        Timezone for the dataframe indicies.
    file_ext, optional: str
        Filetype to specify for reading.

    Returns
    -------
    df: pd.DataFrame
        Contains all fields from files.
    '''
    files = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('txt')]
    df = pd.concat([pd.read_csv(f) for f in files])
    df.index = pd.to_datetime(df['DATE (MM/DD/YYYY)'] +' ' + df['MST'], format='%m/%d/%Y %H:%M')
    df.index = df.index.tz_localize('Etc/GMT+7')
    df.index = df.index.tz_convert('UTC')
    df.index = df.index.tz_convert('US/Mountain')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

def make_pvlib_sys(tilt, elevation, azimuth, lat, lon):
    '''Make LocalizedPVSystem from PVlib.

    Arguments
    ---------
    tilt: float
        Tilt of array.
    elevation: float
        Elevation of system.
    azimuth: float
        Azimuth angle of system.
    lat: float
        Latitutde.
    lon: float
        Longitude.

    Returns
    -------
    sys: LocalizedPVSystem
    '''
    sys_no_loc = pvlib.pvsystem.PVSystem(surface_tilt=tilt, surface_azimuth=azimuth)
    sys_loc = pvlib.location.Location(lat, lon, altitude=elevation)
    sys = pvlib.pvsystem.LocalizedPVSystem(pvsystem=sys_no_loc, location=sys_loc)
    return sys

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From SKLearn docs.
    '''
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 4)

    fig, ax = plt.subplots(figsize=(3,3))
    p = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(p, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    return ax

# def set_up_run(df, ghi_col='GHI', cs_col='GHI Clearsky', ratio_col='ratio', error_col='error'):
#     new_df = pd.DataFrame()
#     new_df.index = df.index
#     new_df[ghi_col] = df[ghi_col]
#     new_df[cs_col] = df[cs_col]
#     new_df[

def calc_ratio(ser1, ser2):
    '''Calculate ratio between two columns.  Invalid values replaced with zero.

    Arguments
    ---------
    ser1: pd.Series
        Numerator.
    ser2: pd.Series
        Denominator.

    Returns
    -------
    ser: pd.Series
        Ratio of ser1 and ser2.
    '''
    ser = (ser1 / ser2).replace([-np.inf, np.inf, np.nan], 1)
    # ser = (ser1 / ser2).replace([-np.inf, np.inf, np.nan], 0)
    ser = pd.Series(ser, index=ser1.index)
    return ser

def calc_abs_ratio_diff(ser1, ser2):
    '''Absolute difference of the ratio of two series from 1.

    Arguments
    ---------
    ser1: pd.Series
        Numerator.
    ser2: pd.Series
        Denominator.

    Returns
    -------
    ser: pd.Series
        |1 - (ser1 / ser2)|.
    '''
    ratio = calc_ratio(ser1, ser2)
    ser = np.abs(1 - ratio)
    ser = pd.Series(ser, index=ser1.index)
    return ser

def mean_abs_diff(this, that):
    '''Calculate mean difference between two series.

    Arguments
    ---------
    this: array-like
    that: array-like

    Returns
    -------
    mad: float
        Mean absolute difference.

    '''
    mad = np.mean(np.abs((this - that)))
    return mad

def metric_calc(df, window, dx, metric_col, model_col):
    '''Calculates window-based metrics on dataframes.  These metrics are used for ML detection.

    This function assumes that key 'abs_diff_ratio stat' exists.  This is the absolute difference
    from 1 of the ratio of GHI: Clearsky GHI stat (clearksy GHI calculated using statistical method).

    This function should be rewritten to be more general (not assume statistical CS, etc).

    Arguments
    ---------
    df: pd.DataFrame
        Dataframe to calculate metrics on.
    window: int
        Size of window for calculating metrics.  Defined as the number of timesteps to use in the window.
        For example, if data is 30 min frequency and window=3, the window will span 1 hour.
    dx: float
        Data frequency in minutes (used for gradient calculation).

    Returns
    -------
    df: pd.DataFrame
        Original dataframe with new columns added.
    '''
    df['metric mean'] = df[metric_col].rolling(window, center=True).mean().fillna(0)
    df['metric std']  = df[metric_col].rolling(window, center=True).std().fillna(0)
    df['metric max']  = df[metric_col].rolling(window, center=True).max().fillna(0)
    df['metric min']  = df[metric_col].rolling(window, center=True).min().fillna(0)
    df['metric range']  = df['metric max'] - df['metric min']

    df['metric grad']      = np.abs(np.gradient(df[metric_col], 30))
    df['metric grad mean'] = df['metric grad'].rolling(window, center=True).mean().fillna(0)
    df['metric grad std']  = df['metric grad'].rolling(window, center=True).std().fillna(0)
    df['metric grad max']  = df['metric grad'].rolling(window, center=True).max().fillna(0)
    df['metric grad min']  = df['metric grad'].rolling(window, center=True).min().fillna(0)
    df['metric grad range'] = df['metric grad max'] - df['metric grad min']

    df['GHI grad']      = np.abs(np.gradient(df['GHI'], dx))
    df['GHI grad mean'] = df['GHI grad'].rolling(window, center=True).mean().fillna(0)
    df['GHI grad std']  = df['GHI grad'].rolling(window, center=True).std().fillna(0)
    df['GHI grad max']  = df['GHI grad'].rolling(window, center=True).max().fillna(0)
    df['GHI grad min']  = df['GHI grad'].rolling(window, center=True).min().fillna(0)
    df['GHI grad range'] = df['GHI grad max'] - df['GHI grad min']

    df['model grad']      = np.abs(np.gradient(df[model_col], dx))
    df['model grad mean'] = df['model grad'].rolling(window, center=True).mean().fillna(0)
    df['model grad std']  = df['model grad'].rolling(window, center=True).std().fillna(0)
    df['model grad max']  = df['model grad'].rolling(window, center=True).max().fillna(0)
    df['model grad min']  = df['model grad'].rolling(window, center=True).min().fillna(0)
    df['model grad range'] = df['model grad max'] - df['model grad min']

    return df


def fit_model(clf, df, cols):
    '''Fit classifier.  Will plot confusion matrix and print CV scores.

    Arguments
    ---------
    clf: sklearn classifier object
        Can be similar, but must have fit and predict member functions.
    df: pd.DataFrame
        Data to learn on.
    cols: list-like
        Columns for training/testing.  It is assumed the last column is the target values.

    Returns
    -------
    clf: sklearn classifier object
        Fitted classifier.
    '''
    X = df[cols[:-1]].values
    y = df[cols[-1]].astype(int).values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    scores = model_selection.cross_val_score(estimator=clf, X=X_train, y=y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_score = metrics.accuracy_score(y_test, y_pred)
    print('CV scoring: {} +/ -  {}'.format(np.round(np.mean(scores), 4), np.round(np.std(scores), 4)))
    print('Test score: {}'.format(test_score))
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, ['cloudy', 'clear'])
    return clf

def fit_model_cv_grid(clf, df, cols, param_grid):
    '''Perform GridSearchCV on classifier.  Will print confusion matrix of optimal
    parameters.

    Arguments
    ---------
    clf: sklearn classifier object
        Can be similar, but must have fit and predict member functions.
    df: pd.DataFrame
        Data to learn on.
    cols: list-like
        Columns for training/testing.  It is assumed the last column is the target values.

    Returns
    -------
    clf_cv: fitted GridSearchCV object
    '''
    X = df[cols[:-1]].values
    y = df[cols[-1]].astype(int).values
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    clf_cv = model_selection.GridSearchCV(clf, param_grid)
    clf_cv.fit(X_train, y_train)
    y_pred = clf_cv.predict(X_test)
    # test_score = metrics.accuracy_score(y_test, y_pred)
    conf_mat = metrics.confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat, ['cloudy', 'clear'])
    return clf_cv

# def metric_calc_model(df, window, dx):
#     '''Calculates window-based metrics on dataframes.  These metrics are used for ML detection.
#
#     This function assumes that key 'abs_diff_ratio stat' exists.  This is the absolute difference
#     from 1 of the ratio of GHI: Clearsky GHI stat (clearksy GHI calculated using statistical method).
#
#     This function should be rewritten to be more general (not assume statistical CS, etc).
#
#     Arguments
#     ---------
#     df: pd.DataFrame
#         Dataframe to calculate metrics on.
#     window: int
#         Size of window for calculating metrics.  Defined as the number of timesteps to use in the window.
#         For example, if data is 30 min frequency and window=3, the window will span 1 hour.
#     dx: float
#         Data frequency in minutes (used for gradient calculation).
#
#     Returns
#     -------
#     df: pd.DataFrame
#         Original dataframe with new columns added.
#     '''
#     df['abs_diff_ratio window mean'] = df['abs_diff_ratio'].rolling(window, center=True).mean().fillna(0)
#     df['abs_diff_ratio window std'] = df['abs_diff_ratio'].rolling(window, center=True).std().fillna(0)
#     df['abs_diff_ratio window max'] = df['abs_diff_ratio'].rolling(window, center=True).max().fillna(0)
#     df['abs_diff_ratio window min'] = df['abs_diff_ratio'].rolling(window, center=True).min().fillna(0)
#
#     df['abs_diff_ratio grad'] = np.abs(np.gradient(df['abs_diff_ratio'], 30))
#     df['abs_diff_ratio grad mean'] = df['abs_diff_ratio grad'].rolling(window, center=True).mean().fillna(0)
#     df['abs_diff_ratio grad std'] = df['abs_diff_ratio grad'].rolling(window, center=True).std().fillna(0)
#     df['abs_diff_ratio grad max'] = df['abs_diff_ratio grad'].rolling(window, center=True).max().fillna(0)
#     df['abs_diff_ratio grad min'] = df['abs_diff_ratio grad'].rolling(window, center=True).min().fillna(0)
#
#     df['GHI grad'] = np.gradient(df['GHI'], dx)
#     df['GHI grad mean'] = df['GHI grad'].rolling(window, center=True).mean().fillna(0)
#     df['GHI grad std'] = df['GHI grad'].rolling(window, center=True).std().fillna(0)
#     df['GHI grad max'] = df['GHI grad'].rolling(window, center=True).max().fillna(0)
#     df['GHI grad min'] = df['GHI grad'].rolling(window, center=True).min().fillna(0)
#
#     df['Clearsky GHI grad'] = np.gradient(df['Clearsky GHI'], dx)
#     df['Clearsky GHI grad mean'] = df['Clearsky GHI grad'].rolling(window, center=True).mean().fillna(0)
#     df['Clearsky GHI grad std'] = df['Clearsky GHI grad'].rolling(window, center=True).std().fillna(0)
#     df['Clearsky GHI grad max'] = df['Clearsky GHI grad'].rolling(window, center=True).max().fillna(0)
#     df['Clearsky GHI grad min'] = df['Clearsky GHI grad'].rolling(window, center=True).min().fillna(0)
#
#     return df

# def parse_kg_climate_class(lat, lon, filename='./Koeppen-Geiger-ASCII.txt'):
#     kg_df = pd.read_csv(filename, delim_whitespace=True)
