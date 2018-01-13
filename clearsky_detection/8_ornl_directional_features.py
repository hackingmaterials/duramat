
# coding: utf-8

# # Table of Contents
#  <p><div class="lev1 toc-item"><a href="#Ground-predictions" data-toc-modified-id="Ground-predictions-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Ground predictions</a></div><div class="lev2 toc-item"><a href="#PVLib-Clearsky" data-toc-modified-id="PVLib-Clearsky-11"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>PVLib Clearsky</a></div><div class="lev1 toc-item"><a href="#Train/test-on-NSRDB-data-to-find-optimal-parameters" data-toc-modified-id="Train/test-on-NSRDB-data-to-find-optimal-parameters-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Train/test on NSRDB data to find optimal parameters</a></div><div class="lev2 toc-item"><a href="#Default-classifier" data-toc-modified-id="Default-classifier-21"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Default classifier</a></div><div class="lev2 toc-item"><a href="#Gridsearch" data-toc-modified-id="Gridsearch-22"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Gridsearch</a></div><div class="lev2 toc-item"><a href="#Best-recall-model" data-toc-modified-id="Best-recall-model-23"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Best recall model</a></div><div class="lev2 toc-item"><a href="#Best-accuracy-model" data-toc-modified-id="Best-accuracy-model-24"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Best accuracy model</a></div><div class="lev2 toc-item"><a href="#Best-precision-model" data-toc-modified-id="Best-precision-model-25"><span class="toc-item-num">2.5&nbsp;&nbsp;</span>Best precision model</a></div><div class="lev2 toc-item"><a href="#Best-f1-model" data-toc-modified-id="Best-f1-model-26"><span class="toc-item-num">2.6&nbsp;&nbsp;</span>Best f1 model</a></div><div class="lev1 toc-item"><a href="#Train-on-all-NSRDB-data,-test-various-freq-of-ground-data" data-toc-modified-id="Train-on-all-NSRDB-data,-test-various-freq-of-ground-data-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Train on all NSRDB data, test various freq of ground data</a></div><div class="lev2 toc-item"><a href="#30-min-freq-ground-data" data-toc-modified-id="30-min-freq-ground-data-31"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>30 min freq ground data</a></div><div class="lev2 toc-item"><a href="#15-min-freq-ground-data" data-toc-modified-id="15-min-freq-ground-data-32"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>15 min freq ground data</a></div><div class="lev2 toc-item"><a href="#10-min-freq-ground-data" data-toc-modified-id="10-min-freq-ground-data-33"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>10 min freq ground data</a></div><div class="lev2 toc-item"><a href="#5-min-freq-ground-data" data-toc-modified-id="5-min-freq-ground-data-34"><span class="toc-item-num">3.4&nbsp;&nbsp;</span>5 min freq ground data</a></div><div class="lev2 toc-item"><a href="#1-min-freq-ground-data" data-toc-modified-id="1-min-freq-ground-data-35"><span class="toc-item-num">3.5&nbsp;&nbsp;</span>1 min freq ground data</a></div><div class="lev1 toc-item"><a href="#Save-model" data-toc-modified-id="Save-model-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save model</a></div><div class="lev1 toc-item"><a href="#Conclusion" data-toc-modified-id="Conclusion-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Conclusion</a></div>

# In[1]:


import pandas as pd
import numpy as np
import os
import datetime
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn import tree
from sklearn import ensemble

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
import cufflinks as cf
cf.go_offline()
init_notebook_mode(connected=True)

from IPython.display import Image


# # Ground predictions

# ## PVLib Clearsky

# Only making ground predictions using PVLib clearsky model and statistical model.  NSRDB model won't be available to ground measurements.

# In[2]:


nsrdb = cs_detection.ClearskyDetection.read_pickle('ornl_nsrdb_1.pkl.gz')
nsrdb.df.index = nsrdb.df.index.tz_convert('EST')
nsrdb.time_from_solar_noon('Clearsky GHI pvlib', 'tfn')


# In[3]:


len(nsrdb.df)


# # Train/test on NSRDB data to find optimal parameters

# ## Default classifier

# In[4]:


train = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
train.trim_dates('01-01-2010', '01-01-2015')
test = cs_detection.ClearskyDetection(nsrdb.df, scale_col=None)
test.trim_dates('01-01-2015', None)


# In[5]:


train.scale_model('GHI', 'Clearsky GHI pvlib', 'sky_status')


# In[6]:


clf = ensemble.RandomForestClassifier(random_state=42)


# In[7]:


utils.calc_all_window_metrics(train.df, 3, meas_col='GHI', model_col='Clearsky GHI pvlib', overwrite=True)


# In[8]:


feature_cols = [
    'tfn',
    'abs_ideal_ratio_diff grad',
    'abs_ideal_ratio_diff grad mean',
    'abs_ideal_ratio_diff grad std',
    'abs_ideal_ratio_diff grad second',
    'abs_ideal_ratio_diff grad second mean',
    'abs_ideal_ratio_diff grad second std',
    'GHI Clearsky GHI pvlib line length ratio',
    'GHI Clearsky GHI pvlib ratio',
    'GHI Clearsky GHI pvlib ratio mean',
    'GHI Clearsky GHI pvlib ratio std',
    'GHI Clearsky GHI pvlib diff',
    'GHI Clearsky GHI pvlib diff mean',
    'GHI Clearsky GHI pvlib diff std'
]

target_cols = ['sky_status']


# In[9]:


import seaborn as sns


# In[10]:


pplot = sns.pairplot(train.df[feature_cols + target_cols], hue='sky_status')


pplot.savefig('pplot.png')
