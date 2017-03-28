

import pandas as pd
import numpy as np
import scipy
import statsmodels.api as smapi


def get_outlier_indices_zscore(series):
    '''
    returns indices of a series where outliers exist based on zscore

    arguments:
        series (pd.Series)

    returns:
        array of indices
    '''
    this_site = series.dropna()
    this_site = pd.DataFrame(this_site, columns=[this_site.name])
    this_site['zscore'] = calc_zscore(this_site)
    indices = this_site[np.abs(this_site['zscore']) > 3].index
    return indices


# def get_outlier_indices_mod_zscore(series):
#     '''
#     returns indices of a series where outliers exist based on
#     modified zscore
#     arguments:
#         series (pd.Series)
#
#     returns:
#         array of indices
#     '''
#     this_site = series.dropna()
#     this_site = pd.DataFrame(this_site, columns=['data'])
#     this_site['mod_zscore'] = calc_modified_zscore(series)
#     indices = this_site[np.abs(this_site['mod_zscore']) > 3.5].index
#     return indices


def calc_zscore(series):
    '''
    calculate zscore of each element of a series using scipy

    aguments:
        series (pd.Series)

    returns:
        array with zscore corresponding to each element of series
    '''
    this_site = series.dropna().values
    return scipy.stats.zscore(this_site)


# def calc_modified_zscore(series):
#     '''
#     returns indices of a series where outliers exist based on
#     modified zscore (http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm)
#
#     m_i = 0.6745 * (x_i - median(x)) / median(abs(x_i - median(x)))
#     any absolute m_i > 3.5 is considered an outlier
#
#     aguments:
#         series (pd.Series)
#
#     returns:
#         array with zscore corresponding to each element of series
#     '''
#     this_site = series.dropna().values
#     this_site = 0.6745 * (this_site - np.median(this_site))
#     this_site /= np.median(np.abs(this_site - np.median(this_site)))
#     return this_site


def longest_streak_indices(series, sample_time='D'):
    '''
    find the longest consecutive streak of non-NaN in a pandas series
    as of now, only finds the first longest instance

    arguments:
        series (pd.Series)
            indices of series should be time
        sampe_time (str)
            frequency of the time series entries

    returns:
        start index, stop index
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

    return index_start, index_stop


def ts_numeric_ratio(series):
    '''
    calculate the ratio of numeric values to non-numeric values (nan, inf)

    arguments:
        series (pd series)
            indices of series should be time
        sampe_time (str)
            frequency of the time series entries

    returns:
        float of # numeric values / # of all values in between first and last numeric values
    '''
    # only consider 'internal' data -- exclude any padded nan/inf values
    # padding can happen when series are stored together in a dataframe with different
    # time ranges
    this_site = series[series.first_valid_index(): series.last_valid_index()]
    return len(this_site.dropna()) / len(this_site)


def get_out_of_range_indices(series, low=-np.inf, low_inclusive=True,
                                     high=np.inf, high_inclusive=True):
    '''
    returns indices of values that are outside of the low, high range

    arguments:
        series (pd.Series)
        low (float)
            low limit of values
        low_inclusive (bool=True)
            True  = [low...high
            False = (low...high
        high (float)
            high limit of values
            True  = low...high]
            False = low...high)

    returns:
        indices of values that are outside the specified range
    '''
    if low > high:
        low, high = high, low

    if low_inclusive:
        if high_inclusive:
            return series[(series < low) | (series > high)].index
        else:
            return series[(series < low) | (series >= high)].index
    else:
        if high_inclusive:
            return series[(series <= low) | (series > high)].index
        else:
            return series[(series <= low) | (series >= high)].index


def ts_ols(series):
    '''
    perform ordinary least squares fit on time series data

    arguments:
        series (pd.Series)

    returns:
        dictionary of results
            b: slope of linear fit (in )
    '''

    this_site = series.dropna()

    # reression always in terms of days
    X = (this_site.index - this_site.index[0]).days
    X = smapi.add_constant(X)

    # build and fit model
    model = smapi.OLS(this_site.values, exog=X, hasconst=True)
    model = model.fit()

    # gather OLS parameters, errors
    b, m = model.params
    rsquared = model.rsquared
    rmse = np.sqrt(model.mse_resid)

    output = {'m': m, 'b': b, 'rsquared': rsquared, 'rmse': rmse, 'model': model}
    return output


def calc_degredation_rate_ols(series):
    '''
    calculate degradation rate of time series data using OLS
    this function could be modified to use other methods of calculated degradation rate
    and serve more as a wrapper function

    arguments:
        series (pd.Series)

    returns:
        dictionary of OLS results and degradation rate
    '''
    ols_result = ts_ols(series)

    # calcualte degradation rate per year  b / m * 365 * 100
    deg_rate = ols_result['m'] / ols_result['b'] * 100 * 365
    ols_result['deg_rate'] = deg_rate

    return ols_result


def ts_csd(series):
    '''
    perorm classical seasonal decomposition on a time-series

    arguments:
        series (pd.Series)
            series is sampled monthly

    returns:
        fill in
    '''
    # resample data on monthly basis
    # (daily and weekly works too - can add options)
    # use linear interpolation to fill in missing values
    # csd cannot have missing values
    this_site = series[series.first_valid_index(): series.last_valid_index()].\
                resample('D').mean().interpolate()

    result = smapi.tsa.seasonal_decompose(this_site, freq=365)

    # build dictionary of output
    output = {'seasonal': result.seasonal,
              'trend': result.trend,
              'residual': result.resid}

    return output


def calc_degredation_rate_csd(series):
    '''
    calculate degradation rate of time series data
    first a csd is performed to extract the overall trend of the data
    then an OLS regression is performed on the trend and used to compute
    percent degradation

    arguments:
        series (pd.Series)

    returns:
        dictionary of OLS results and degradation rate
    '''
    # perform csd to get
    # seasonality, trend, and residual factors
    csd_result = ts_csd(series)

    # perform ols to get linear estimate of csd trend
    return calc_degredation_rate_ols(csd_result['trend'])


def ts_lowess(series):
    '''
    perform LOcally WEighted Scatterplot Smoothing on time series

    arguments:
        series (pd.Series)

    returns:
        pd.Series of lowess datapoints
    '''
    # interpolate missing values - could also drop but smoothness of
    # fit would suffer
    this_site = series[series.first_valid_index(): series.last_valid_index()].\
                resample('D').mean().interpolate()

    # perform regression and return as time series
    result = smapi.nonparametric.lowess(this_site.values, this_site.index)
    result = pd.Series(result[:, 1], index=this_site.index)
    return result


def calc_degredation_rate_lowess(series):
    '''
    calculate degradate rate of time series data
    first, a lowess fit is generated on a given series
    then an OLS regression is performed on the lowess fit
    to compute percent degradation
    '''
    # perform lowess regression
    lowess_result = ts_lowess(series)

    # perform ols to get linear estimate of trend
    return calc_degredation_rate_ols(lowess_result)
