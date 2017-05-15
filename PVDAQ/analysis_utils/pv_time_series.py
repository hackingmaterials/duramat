

import pandas as pd
import numpy as np
import scipy
import statsmodels.api as smapi


def get_outlier_indices_zscore(series, cutoff=3):
    '''Returns indices of a series where |zscore| > cutoff.

    Arguments
    ---------
    series: pd.Series
        Data for which outlier indices will be found and returned.
    cutoff: float
        Limit of zscore cutoff.  Zscores greater than this value are outliers.

    Returns
    -------
    series: pd.Series of indices
    '''
    this_site = series.dropna()
    # this_site = pd.DataFrame(this_site, columns=[this_site.name])
    zscore = pd.Series(calc_zscore(this_site))
    zscore.index = this_site.index
    return zscore[np.abs(zscore) > cutoff].index


def calc_zscore(series):
    '''Calculate zscore of each element of a series using scipy.

    Arguments
    ---------
    series: pd.Series
        Data to calculate zscore on.

    Returns
    -------
    series: pd.Series of calculated zscore
    '''
    this_site = series.dropna().values
    return scipy.stats.zscore(this_site)


def longest_streak_indices(series, sample_time='D'):
    '''Find the longest consecutive streak of non-NaN in a series.

    Right now, only finds first longest instance.

    Arguments
    ---------
    series: pd.Series
        Time series data.
    sample_time: str
        Frequency of the time series entries.

    Returns
    -------
    start_index: pd.Series timeindex
        Index of first element in longest streak.
    stop_index: pd.Series timeindex
        Index of final element in longest streak.
    '''
    time_delta = pd.Timedelta(1, sample_time)

    # trim nans that may be on either side of the data
    tmp_series = series[series.first_valid_index(): series.last_valid_index()]
    if len(tmp_series) == 1:
        return tmp_series.index[0], tmp_series.index[0]
    elif len(tmp_series) == 0:
        raise ValueError('Received empty series')

    # find locations of nan values
    nan_loc = tmp_series[pd.isnull(tmp_series)]

    if len(nan_loc) == 0:
        return tmp_series.index[0], tmp_series.index[-1]

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


def ts_numeric_ratio(series, include_inf=False):
    '''Calculate the ratio of numeric values to non-numeric values (nan, inf).

    Only consides "internal" nan/inf by trimming padded nan values.

    Arguments
    ---------
    series: pd.Series
        Time series data.

    Returns
    -------
    ratio: float
        Ratio of numeric data to total length of data.
    '''
    # this_site = series[series.first_valid_index(): series.last_valid_index()]
    # ratio = len(this_site.dropna()) / len(this_site)
    if include_inf:
        this_site = series
    else:
        this_site = series.replace([np.inf, -np.inf], np.nan)
    ratio = len(this_site.dropna()) / len(this_site)
    return ratio


def get_out_of_range_indices(series, low=-np.inf, low_inclusive=True, high=np.inf, high_inclusive=True):
    '''Returns indices of values that are outside of the low, high range in a series.

    Arguments
    ---------
    series: pd.Series
        Data to inspect.
    low: float
        low limit of values
    low_inclusive: bool
        True  = [low...high
        False = (low...high
    high: float
        high limit of values
    high_inclusive: bool
        True  = low...high]
        False = low...high)

    Returns
    -------
    pd.Series indices
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


def ts_data_quality(this_site, edit_indices=None, edit_val=np.nan):
    '''Caclulates data quality metrics of time series data.

    The function will calculate the percentage of non-NaN data, the span of the
    data (in days, weeks, months, and years), and the longest consecutive streak of datapoints
    (in days, weeks, months, and years).  These metrics can be used to cutoff small data
    sets or data sets with many missing values.

    This function can also edit certain indices to be a certain value for the purpose
    of removing outliers or restricting data be be in a certain range, etc.

    Arguments
    ---------
    this_site: pd.Series
        Time series data.
    edit_indices: pd.Index
        Indices of values to edit.
    edit_val: numeric
        The value to substitute at edit_indices.

    Returns
    -------
    quality: pd.Series
        Quality of data that gives percent of data that is non-NaN, span of data, and longest
        consecutive non-NaN streak of data.
    '''
    this_site = this_site[this_site.first_valid_index(): this_site.last_valid_index()].copy()
    # print(this_site.name, len(this_site))
    if edit_indices is not None:
        this_site[[x for x in edit_indices if x in this_site.index]] = edit_val
    if this_site.dropna().empty:
        pct, span, longest_streak = 0, 0, 0
    else:
        pct = ts_numeric_ratio(this_site)
        span = (this_site.index[-1] - this_site.index[0]).days
        start, stop = longest_streak_indices(this_site, sample_time='D')
        longest_streak = (stop - start).days + 1
    quality = pd.Series([pct, span, span / 7, span / 30.5, span / 365,
                        longest_streak, longest_streak / 7, longest_streak / 30.5, longest_streak / 365],
                        index=['pct', 'span (D)', 'span (W)', 'span (M)', 'span (Y)',
                               'longest_streak (D)', 'longest_streak (W)',
                               'longest_streak (M)', 'longest_streak (Y)'], name=this_site.name)
    return quality


def df_data_quality(df, indices_filter_method=None, edit_val=np.nan, *args, **kwargs):
    '''Caclulates quality of mulitple time series data sets stored in a dataframe.

    The data quality can be calculated for multiple time series datasets here (stored
    in a dataframe).  With this function, it is also possible to select a function to
    filter indices (e.g. get_outlier_indices_zscore, get_out_of_range_indices).

    Arguments
    ---------
    df: pd.DataFrame
        Assumed to be multiple time series datasets in single dataframe.
    indices_filter_method: function
        Function that returns indices to be changed to edit_val.
    edit_val: numeric
        Value to change indices to (if filter method is used).
    *args, **kwargs:
        Used for passing extra arguments to indices_filter_method if needed.

    Returns
    -------
    quality_df: pd.DataFrame
        The quality of each series is a column in dataframe.
    '''
    series_list = []
    for site in sorted(df.keys()):
        this_site = df[site]
        if this_site.dropna().empty:
            continue
        if indices_filter_method is not None:
            indices = indices_filter_method(this_site, *args, **kwargs)
        else:
            indices = None
        quality = ts_data_quality(this_site, indices, edit_val)
        if np.all(quality.values == 0.):
            continue
        else:
            series_list.append(quality)
    quality_df = pd.concat(series_list, axis=1)
    return quality_df


def ts_ols(series, ols_kwargs={}):
    '''Perform ordinary least squares fit on time series data using statsmodels.

    The slope will be in terms of days.

    Arguments
    ---------
    series: pd.Series
        Time series data on which OLS will be fit to.
    ols_kwargs: dict
        Keyword arguments for OLS call.

    Returns
    -------
    output: dict
        Includes slope (m: float),
                 intercept (b: float),
                 rsquared: float,
                 rmse: float,
                 fit model (model: RegressionResult object)
    '''
    this_site = series.dropna()

    # reression always in terms of days
    X = (this_site.index - this_site.index[0]).days
    X = smapi.add_constant(X)

    # build and fit model
    model = smapi.OLS(this_site.values, exog=X, hasconst=True, **ols_kwargs)
    model = model.fit()

    # gather OLS parameters, errors
    b, m = model.params
    rsquared = model.rsquared
    rmse = np.sqrt(model.mse_resid)

    output = {'m': m, 'b': b, 'rsquared': rsquared, 'rmse': rmse, 'model': model}
    return output


def calc_degredation_rate_ols(series, ols_kwargs={}):
    '''Calculate yearly degradation (percent change) rate of time series data using OLS.

    Degradation rate is calculated as follows:
        100 * (m / b) *  365

    Where m and b are the slope and intercept from an OLS, respectively.  Slope is always
    calculated in terms of days.  Degradation rate is calculated for yearly basis.

    Arguments
    ---------
    series: pd.Series
        Time series data.
    ols_kwargs: dict
        Keyword arguments for ordinary least squares.

    Returns
    -------
    ols_result: dict
        Adds degradation rate (deg_rate: float) to result of ts_ols function.
    '''
    ols_result = ts_ols(series, ols_kwargs=ols_kwargs)

    # calcualte degradation rate per year  b / m * 365 * 100
    deg_rate = ols_result['m'] / ols_result['b'] * 100 * 365
    ols_result['deg_rate'] = deg_rate

    return ols_result


def ts_seasonal_decomp(series, resample_time='D', sd_kwargs={}):
    '''Perform classical seasonal decomposition on time series.

    Decomposition uses statsmodels.  Series will be resampled (using mean).
    NaN values raise errors with CSD, so missing values are interpolated (linearly for now).


    Arguments
    ---------
    series: pd.Series
        Time series to be seasonally decomposed.
    resample_time: str
        How to resample the time series (D=daily, W=weekly, M=monthly).
    sd_kwargs: dict
        Keyword arguments for seasonal decomposition.

    Returns
    -------
    output: dict
        Contains CSD results, which is the seasonal, trend, and residual values.
        Each is a numpy array.
    '''
    resample_time = resample_time.upper()
    trend_freq_dict = {'D': 365, 'W': 52, 'M': 12}
    if resample_time not in ('D', 'W', 'M'):
        raise NotImplementedError('Only daily (D), weekly (W), and monthly (M) supported now.')
    trend_freq = trend_freq_dict[resample_time]

    this_site = series[series.first_valid_index(): series.last_valid_index()].\
                resample(resample_time).mean().interpolate()

    result = smapi.tsa.seasonal_decompose(this_site, freq=trend_freq, **sd_kwargs)

    # build dictionary of output
    output = {'seasonal': result.seasonal,
              'trend': result.trend,
              'residual': result.resid}

    return output


def calc_degredation_rate_seasonal_decomp(series, resample_time='D', sd_kwargs={}, ols_kwargs={}):
    '''Calculate degradation rate of on the trend of a seasonally decomposed time series.

    Arguments
    ---------
    series: pd.Series
        Time series data.
    resample_time: str
        How to resample the time series (D=daily, W=weekly, M=monthly).
    sd_kwargs:
        Keyword arguments for seasonal decomposition.
    ols_kwargs:
        Keyword arguments for ordinary least squares decomposition.

    Returns
    -------
    result: dict
        Result of ts_ols performed on the trend of a CSD.
    '''
    # perform csd to get
    # seasonality, trend, and residual factors
    csd_result = ts_seasonal_decomp(series, resample_time=resample_time, sd_kwargs=sd_kwargs)

    # perform ols to get linear estimate of csd trend
    result = calc_degredation_rate_ols(csd_result['trend'], ols_kwargs)
    return result
    # return calc_degredation_rate_ols(csd_result['trend'])


def ts_lowess(series, resample_time='D', lowess_kwargs={}):
    '''Perform LOcally WEighted Scatterplot Smoothing on time series.

    Missing values must be interpolated (right now this is done linearly).

    Arguments
    ---------
    series: pd.Series
        Time series data.
    resample_time: str
        How to resample the time series (D=daily, W=weekly, M=monthly).
    lowess_kwargs: dict
        Keyword arguments for LOWESS.

    Returns
    -------
    result: pd.Series:
        Time series of LOWESS points.
    '''
    # interpolate missing values - could also drop but smoothness of
    # fit would suffer
    this_site = series[series.first_valid_index(): series.last_valid_index()].\
                resample(resample_time).mean().interpolate()

    # perform regression and return as time series
    result = smapi.nonparametric.lowess(this_site.values, this_site.index, **lowess_kwargs)
    result = pd.Series(result[:, 1], index=this_site.index)
    return result


def calc_degredation_rate_lowess(series, resample_time='D', lowess_kwargs={}, ols_kwargs={}):
    '''Calculate degradate rate of LOWESS trend for time series data.

    Arguments
    ---------
    series: pd.Series
        Time series data.
    resample_time: str
        How to resample the time series (D=daily, W=weekly, M=monthly).
    lowess_kwargs: dict
        Keyword arguments for LOWESS.
    ols_kwargs: dict
        Keyword arguments for OLS.

    Returns
    -------
    ols_result: dict
        Result of of calc_degradation_rate_ols done on the LOWESS trend.
    '''
    # perform lowess regression
    lowess_result = ts_lowess(series, resample_time, lowess_kwargs)

    # perform ols to get linear estimate of trend
    ols_result = calc_degredation_rate_ols(lowess_result, ols_kwargs)
    return ols_result
    # return calc_degredation_rate_ols(lowess_result)


