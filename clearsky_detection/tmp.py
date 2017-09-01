
import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree

import pytz
import itertools
import visualize
import utils
import pydotplus
import xgboost as xgb

from sklearn import metrics
from sklearn import model_selection

import pvlib
import cs_detection

import visualize_plotly as visualize
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

nsrdb = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('EST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')

ground = cs_detection.ClearskyDetection.read_pickle('ornl_ground_1.pkl.gz')
ground.df.index = ground.df.index.tz_convert('EST')

feature_cols = [
'tfn',
'abs_ideal_ratio_diff',
'abs_ideal_ratio_diff mean',
'abs_ideal_ratio_diff std',
'abs_ideal_ratio_diff max',
'abs_ideal_ratio_diff min',
'GHI Clearsky GHI pvlib gradient ratio',
'GHI Clearsky GHI pvlib gradient ratio mean',
'GHI Clearsky GHI pvlib gradient ratio std',
'GHI Clearsky GHI pvlib gradient ratio min',
'GHI Clearsky GHI pvlib gradient ratio max',
'GHI Clearsky GHI pvlib gradient second ratio',
'GHI Clearsky GHI pvlib gradient second ratio mean',
'GHI Clearsky GHI pvlib gradient second ratio std',
'GHI Clearsky GHI pvlib gradient second ratio min',
'GHI Clearsky GHI pvlib gradient second ratio max',
'GHI Clearsky GHI pvlib line length ratio',
'GHI Clearsky GHI pvlib line length ratio gradient',
'GHI Clearsky GHI pvlib line length ratio gradient second'
]

target_cols = ['sky_status']

train = cs_detection.ClearskyDetection(nsrdb.df)
train.trim_dates('01-01-2010', '06-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('06-01-2015', None)

train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')

utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)
param_grid = {'max_depth': [3, 4, 5], 'n_estimators': [200, 300, 400], 'learning_rate': [.1, .01, .001]}

clf = xgb.XGBClassifier()
clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())

pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=False)
score = metrics.f1_score(test.df['sky_status'], pred)
print(score)

test = cs_detection.ClearskyDetection(nsrdb.df)
test.trim_dates('06-01-2015', None)

pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True)
score = metrics.f1_score(test.df['sky_status'], pred)
print(score)

import itertools
import warnings
# with warnings.catch_warnings():
    # warnings.simplefilter("ignore")
best_score = 0
best_params = {}
# for depth, n_est, lr in itertools.product(param_grid['max_depth'], param_grid['n_estimators'], param_grid['learning_rate']):
for depth in param_grid['max_depth']:
    for n_est in param_grid['n_estimators']:
        for lr in param_grid['learning_rate']:
            clf = xgb.XGBClassifier(max_depth=depth, n_estimators=n_est, learning_rate=lr, n_jobs=4)
            clf.fit(train.df[feature_cols].values, train.df[target_cols].values.flatten())
            print('fit')
            # test = cs_detection.ClearskyDetection(nsrdb.df)
            # test.trim_dates('01-01-2015', None)
            pred = test.iter_predict_daily(feature_cols, 'GHI', 'Clearsky GHI pvlib', clf, 3, multiproc=True)
            print('pred')
            score = metrics.f1_score(test.df['sky_status'], pred)
            indicator = ''
            if score > best_score:
                best_score = score
                best_params['max_depth'] = depth
                best_params['n_estimators'] = n_est
                best_params['learnin_rate'] = lr
                indicator = '*'
            print('max_depth: {}, n_estimators: {}, learning_rate: {}, accuracy: {} {}'.format(depth, n_est, lr, score, indicator))
