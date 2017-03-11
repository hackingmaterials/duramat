'''
utility functions used in jupyter notebooks
'''

import scipy
import pandas as pd
import numpy as np
import copy
from sklearn import linear_model

def get_outlier_index_zscore(series):
    '''
    returns indices of a series where outliers exist based on zscore

    arguments:
        series (pandas series)

    returns:
        series of indices
    '''
    this_site = series.dropna()
    this_site = pd.DataFrame(this_site, columns=[this_site.name])
    this_site['zscore'] = scipy.stats.zscore(this_site)
    indices = this_site[np.abs(this_site['zscore']) > 3].index
    return indices


def get_outlier_index_mod_zscore(series):
    '''
    returns indices of a series where outliers exist based on
    modified zscore

    m_i = 0.6745 * (x_i - median(x)) / median(abs(x_i - median(x)))

    any absolute m_i > 3.5 is considered an outlier

    arguments:
        series (pandas series)

    returns:
        series of indices
    '''
    this_site = series.dropna()
    this_site = pd.DataFrame(this_site, columns=['data'])
    this_site['mod_zscore'] = 0.6745 * (this_site['data'] - this_site['data'].median())
    this_site['mod_zscore'] /= np.abs(this_site['data'] - this_site['data'].median()).median()
    indices = this_site[np.abs(this_site['mod_zscore']) > 3.5].index
    return indices



def remove_outliers_zscore(df):
    '''
    returns a copy of df with outliers (of each column) removed using z-scoring
    an outlier is defined as being > 3 std_dev from mean

    *** IMPORTANT ***
    this function is nearly identical to get_outlier_index_zscore but is less flexible.
    it will be removed once the notebooks have been updated.

    arguments:
        df (pandas dataframe)

    returns:
        pandas dataframe (copy)
    '''
    series_list = []
    for site in sorted(df.keys()):
        this_site = pd.DataFrame(df[site].dropna())
        this_site['zscore'] = np.abs(scipy.stats.zscore(this_site))
        this_site = this_site[this_site['zscore'] < 3]
        series = this_site[site]
        series.name = site
        series.index = this_site.index
        series_list.append(series)
    return pd.concat(series_list, ignore_index=False, axis=1)


def remove_outliers_modified_zscore(df):
    '''
    returns a copy of df with outliers (of each column) removed
    using modified z-scoring

    m_i = 0.6745 * (x_i - median(x)) / median(abs(x_i - median(x)))

    any absolute m_i > 3.5 is considered an outlier

    *** IMPORTANT ***
    this function is nearly identical to get_outlier_index_mod_zscore but is less flexible.
    it will be removed once the notebooks have been updated.

    arguments:
        df (pandas dataframe)

    returns:
        pandas dataframe (copy)
    '''
    series_list = []
    for site in sorted(df.keys()):
        this_site = pd.DataFrame(df[site].dropna())
        this_site['zscore'] = 0.6745 * (this_site[site] - this_site[site].median())
        this_site['zscore'] /= np.abs(this_site[site] - this_site[site].median()).median()
        this_site = this_site[np.abs(this_site['zscore']) < 3.5]
        series = this_site[site]
        series.name = site
        series.index = this_site.index
        series_list.append(series)
    return pd.concat(series_list, ignore_index=False, axis=1)


def remove_outliers(df, method):
    '''
    calls either remove outliers function based on preference

    arguments:
        df (pandas dataframe)
        method (str)
            which method to use to remove outliers

    returns:
        pandas dataframe (copy)
    '''
    functions = {'standard': remove_outliers_zscore,
                 'modified': remove_outliers_modified_zscore}
    try:
        return functions[method](df)
    except KeyError('mode must be one of the two options {}'.format(functions.keys())):
        return None


def linear_fit_ts(df):
    '''
    performs columnwise linear regression on df
    df is assumed to be time-series as the indices
    are used as the independent var

    the purpose of this function is a descriptive linear trend, not a
    rigorous predictive model

    arguments:
        df (pandas dataframe)

    returns:
        pandas dataframe
            regression coefficient, intercept, and score
        dictionary
            linear fit models by column
    '''
    regressions = []
    fits = {}

    for site in sorted(df.keys()):
        lin_reg = linear_model.LinearRegression()
        this_site = df[site].dropna()

        X = (this_site.index - this_site.index[0]).days
        X = X.reshape(-1, 1)

        y = this_site.values

        lin_reg.fit(X, y)
        r_sq = lin_reg.score(X, y)

        fits[site] = copy.deepcopy(lin_reg)

        tmp = pd.Series([lin_reg.coef_[0], lin_reg.intercept_, r_sq],
                        index=['coefficient', 'intercept', 'r_sq'],
                        name=site)

        regressions.append(tmp)

    regressions = pd.concat(regressions, axis=1, ignore_index=False)
    return regressions, fits


def predict_linear_fit(xvalues, fit):
    '''
    predict observables from xvalues

    arguments:
        xvalues (numeric or array-like)
            independent variable
        fit (object with predict method)
            predictive model, assumed to be a SKLearn object (or one with predict method)

    returns:
        array-like predictions of observalbes
    '''
    X = (xvalues - xvalues[0]).days.reshape(-1, 1)
    y_pred = fit.predict(X)

    return y_pred


def read_csv_to_ts(site_ids_list, property_of_interest, time_block):
    '''
    read csvs for all sites in site_ids_list, get property_of_interest, and return a pandas
    dataframe where each column is the property of interest for a given site and indices
    are timestamps for the data

    right now it is assumed that the filenames are saved in the directory ./data_sets/
    and the names are of the following form:

    {site_id}-{time_block}.csv
        ^          ^
        |          |_ aggregation time from API call (hourly, daily, weekly, monthly)
        |
        |__ integer system_id

    NOTE: in this directory there is a scraping script that will get data from PVDAQ and save it
    in the above format.  it does not automatically create data_sets/.  Right now it is up to you
    to create the directory and move the files.

    TO DO: change arguments to allow for great flexibility (path to file, supply filenames of any format, etc)

    arguments:
        site_ids_list (array like)
            list of integers for system ids in pvdaq database
        property_of_interest (str)
            column header for property desired
        time_block (str)
            aggregate data time-frame (from pvdaq) can be hourly, daily, weekly, or monthly
    returns:
        pandas dataframe indexed as time-series
    '''
    sites_list = []
    for site in site_ids_list:
        try:
            df = pd.read_csv('data_sets/' + str(site) + '-' + time_block + '.csv')
        except FileNotFoundError:
            continue

        if not property_of_interest in df.keys():
            print('site {} -- {} not in csv'.format(site, property_of_interest))
            continue

        if df[property_of_interest].dropna().empty:
            print('site {} -- {} is empty'.format(site, property_of_interest))
            continue

        series = df[property_of_interest]
        series.index = pd.to_datetime(df['measdatetime'])
        series.name = site
        sites_list.append(series)

    systems_df = pd.concat(sites_list, ignore_index=False, axis=1)
    return systems_df


def calc_pct_deg(regression_info):
    '''
    calculate percent degredation (negative values) or improvate (positive values)
    for linear fit

    arguments:
        regression_info (pandas dataframe)
            contains linear fit information - coefficient and intercept
            linear fit columns are sites, linear fit rows are coeff, intercept, r_sq

    returns:
        pandas series of pct values
    '''
    return pd.Series(100 * regression_info.T['coefficient'] / regression_info.T['intercept'])


def ts_longest_streak(series, sample_time='D'):
    '''
    find the longest consecutive streak of non-NaN in a pandas series
    as of now, only finds the first longest instance

    arguments:
        series (pd series)
            indices of series should be time
        sampe_time (str)
            frequency of the time series entries
    '''
    time_delta = pd.Timedelta(1, sample_time)

    # trim nans that may be on either side of the data
    tmp_series = series[series.first_valid_index(): series.last_valid_index()]

    # find locations of nan values
    nan_loc = tmp_series[tmp_series.isnull()]

    # assume longest streak is start of original series to first nan value
    longest_streak = nan_loc.index[0] - tmp_series.index[0]
    index_start = tmp_series.index[0]
    index_stop = nan_loc.index[0] - time_delta

    # check ranges between nan values
    for i in range(1, len(nan_loc)):
        this_streak = nan_loc.index[i] - nan_loc.index[i - 1]
        if this_streak > longest_streak:
            longest_streak = this_streak
            index_start = nan_loc.index[i - 1] + time_delta
            index_stop = nan_loc.index[i] - time_delta

    # check last nan_value to end of original series
    this_streak = tmp_series.index[-1] - nan_loc.index[-1]
    if this_streak > longest_streak:
        longest_streak = this_streak
        index_start = nan_loc.index[-1] + time_delta
        index_stop = tmp_series.index[-1]

    return series[(series.index >= index_start) & (series.index <= index_stop)]

















