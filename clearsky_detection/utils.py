

import os
import itertools

import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_dir_nsrdb(dir_path, timezone, file_ext='csv'):
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
    df = pd.read_csv(file_w_path, parse_dates=['Date-Time'], index_col=['Date-Time'])
    df.index = df.index.tz_localize(timezone1)
    df.index = df.index.tz_convert(timezone2)
    df = df.sort_index()
    df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')).fillna(0)
    df = df[~df.index.duplicated(keep='first')]
    return df

def read_srrl_dir(dir_path, timezone, file_ext='txt'):
    files = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('txt')]
    df = pd.concat([pd.read_csv(f) for f in files])
    df.index = pd.to_datetime(df['DATE (MM/DD/YYYY)'] +' ' + df['MST'], format='%m/%d/%Y %H:%M')
    df.index = df.index.tz_localize('Etc/GMT+7')
    df.index = df.index.tz_convert('US/Mountain')
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    return df

def make_pvlib_sys(tilt, elevation, azimuth, lat, lon):
    sys_no_loc = pvlib.pvsystem.PVSystem(surface_tilt=tilt, surface_azimuth=azimuth)
    sys_loc = pvlib.location.Location(lat, lon, altitude=elevation)
    sys = pvlib.pvsystem.LocalizedPVSystem(pvsystem=sys_no_loc, location=sys_loc)
    return sys

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    From sklearn docs.
    """
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











