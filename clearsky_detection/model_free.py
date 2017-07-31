
import os
import warnings

import pandas as pd
import numpy as np

from scipy import linalg
from scipy import interpolate


def main():
    # file_path = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    # cols = ['Global_Wm2', 'Date-Time']
    # data = pd.read_csv(file_path, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])

    # data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    # data = data[(data.index >= '2016-07-01') & (data.index < '2016-07-15')]
    # data = pd.Series(data['Global_Wm2'], index=data.index)
    filename = os.path.expanduser('~/data_sets/snl_raw_data/1429_1405/raw_1405_weather_for_1429.csv')
    cols = ['Global_Wm2', 'Date-Time']
    data = pd.read_csv(filename, parse_dates=['Date-Time'], usecols=cols, index_col=['Date-Time'])
    data.index = data.index.tz_localize('Etc/GMT+7')
    data = data.reindex(pd.date_range(start=data.index[0], end=data.index[-1], freq='1min')).fillna(0)
    data = pd.Series(data['Global_Wm2'], index=data.index)
    data = data[(data.index >= '2016-07-01') & (data.index < '2016-07-15')]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mf = ModelFreeDetect(data)

        _ = mf.standard_detection()
        print('standard detection')

        _ = mf.mean_detection()
        print('mean detection')

        _ = mf.democratic_detection()
        print('democratic detection')

        _ = mf.deviation_time_filter()
        print('filtered')

        _ = mf.standard_detection()
        print('filtered standard detection')

        _ = ModelFreeDetect(data)
        _ = mf.deviation_time_filter(model_fxn=np.percentile)
        print('filtered percentile')

        _ = mf.standard_detection()
        print('standard filtered percentile')


class ModelFreeDetect(object):
    """Class used to analyze irradiance curves to determine sky clarity."""

    def __init__(self, data, window=30, copy=True):
        """Initialize class with data.

        Arguments
        ---------
        data: pd.Series
            Time series data.
        window, optional: int
            Number of measurements per window.
        metric_tol, optional: float
            Tolerance for determining clear/cloudy skies.
        copy, optional: bool
            Create copy of data or not.

        """
        if len(pd.unique(data.index.to_series().diff().dropna())) != 1:
            raise NotImplementedError('You must use evenly spaced time series data.')
        if copy:
            self.data = data.copy()
        else:
            self.data = data
        self.data.index = self.data.index.tz_convert('UTC')
        self.data.name = 'data'
        self._dates = self.data.index.date
        # self.window = window
        self.filter_mask = pd.Series(True, index=self.data.index)

    def reset_filter_mask(self, value=True):
        """Resets filter mask to desired state (True/False).

        Arguments
        ---------
        value, optional: bool
            Value for mask to take (True or False).
        """
        if value not in (True, False):
            raise ValueError('Mask value must be bool (True/False).')
        self.filter_mask = pd.Series(value, index=self.data.index)

    def generate_window_slices(self, arr):
        """Generate arrays for slicing data into windows.

        Arguments
        ---------
        arr: np.array

        Returns
        -------
        slices: np.ndarray
            Hankel matrix for slicing data into windows.
        """
        slices = linalg.hankel(np.arange(0, len(arr) - self.window + 1),
                               np.arange(len(arr) - self.window, len(arr)))
        return slices

    def calc_window_integral(self, array):
        """Calculate integral of array values.

        Array values are assumed to be y values and dx is assumed to be one.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        val: float
            Integral of array
        """
        val = np.trapz(array)
        return val

    def calc_window_diff_coeff_variation(self, array):
        """Calculate coefficient of variation of differences for a given window.

        cv = stdev / |mean|

        Arguments
        ---------
        array: np.array

        Returns
        -------
        cv: float
            Coefficient of variation for array.
        """
        y_diff = np.diff(array)
        cv = np.std(y_diff) / np.abs(np.mean(y_diff))
        return cv

    def calc_window_line_length_norm(self, array):
        """Calculate normalizedline length of an array.

        The points are assumed to be evenly spaced (dx=1).
        Line length ar normalized by the
        straight line distances between the first and last array elements.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        line_length_norm: float
            Total length line (as described by array) travels normalized by the endpoint-to-endpoint line length.
        """
        line_length = self.calc_window_line_length(array)
        endpoint_line_length = np.sqrt(np.square(array[-1] - array[0]) + np.square(len(array) - 1))
        line_length_norm = line_length / endpoint_line_length
        return line_length_norm

    def calc_window_line_length(self, array):
        """Calculate line length of an array.  The points are assumed to
        be evenly spaced (dx=1).

        Arguments
        ---------
        array: np.array

        Returns
        -------
        line_length: float
            Total length line (as described by array) travels.
        """
        diffs = np.diff(array)
        line_length = np.sum(np.sqrt(np.square(diffs[:]) + 1))  # 1 for dx
        return line_length

    def get_midval(self, array):
        """Returns element at the midpoint of an array.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        midval: variable
            Element at midpoint of array.
        """
        midpoint = (len(array) // 2)
        midval = array[midpoint]
        return midval

    def calc_pct(self, array, metric_tol):
        """Calculate percent of array elements that are less than or equal to a tolerance.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        pct: float
            Percent of values that are <= self.metric_tol.
        """
        pct = np.sum((array <= metric_tol).astype(int)) / len(array)
        return pct

    def calc_window_avg(self, array, weights=None):
        """Calculate average value of array.  Can points can be weighted.

        Arugments
        ---------
        array: np.array

        Returns
        -------
        val: float
            Mean value of array.
        """
        if weights == 'gaussian':
            center = len(array) // 2
            weights = np.asarray([np.exp(-(i - center)**2 / (2 * 1**2)) for i in range(len(array))])
        else:
            weights = np.ones(len(array))

        return np.average(array, weights=weights)

    def calc_window_max(self, array):
        """Calculate average value of array.

        Arugments
        ---------
        array: np.array

        Returns
        -------
        val: float
            mean value of array
        """
        return np.max(array)

    def calc_window_derivative_std_normed(self, array):
        """Calculate std deviation of derivatives of array.  This is normalized
        by the average value of irradiance over the time interval.  This metric
        is used in Reno-Hansen detection.  Same as coefficient of variation.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        norm_std: float
            Standard devation of derivatives divided by average array value.
        """
        window_mean = self.calc_window_avg(array)
        dy = np.diff(array)  # diff as that's what Reno-Hansen indicate
        std_dy = np.std(dy, ddof=1)  # dx always 1
        return std_dy / window_mean

    def calc_window_derivative_avg(self, array):
        """Calculate average derivative of array.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        val: float
            average rate of change
        """
        val = np.mean(np.gradient(array))
        return val

    def calc_window_derivative_std(self, array):
        """Calculate average derivative of array.

        Arguments
        ---------
        array: np.array

        Returns
        -------
        val: float
            average rate of change
        """
        val = np.std(np.diff(array))
        return val

    def calc_window_max_diff(self, arr1, arr2):
        """Calculate the maximum difference between two arrays slopes.
        Reno-Hansen call this the slope (since dx=1).

        Arugments
        ---------
        arr1: np.array
        arr2: np.array

        Returns
        -------
        max_abs_diff: float
            Maximum absolute difference between two arrays
        """
        meas_diff = np.diff(arr1)
        model_diff = np.diff(arr2)
        max_abs_diff = np.max(np.abs(meas_diff - model_diff))
        return max_abs_diff

    def by_time_of_day_transform(self):
        """Transform self.data time series (spanning several days) into a data frame
        with each row being time of a day and each column as a date.

        Returns
        -------
        by_time: pd.DataFrame
            DataFrame with time of day as rows and dates as columns.
        """
        tmp = pd.DataFrame(self.data, index=self.data.index)
        by_time = pd.pivot_table(tmp, index=tmp.index.time, columns=tmp.index.date, values='data')
        return by_time

    def generate_day_range(self, num_days=30):
        """Generates groups of days for statistical analysis.

        Arguments
        ---------
        window_size, optional: int
            Size of window (in days).

        Yields
        ------
        day_range: tuple
            (Day of interest, date of days (+/- window_size / 2))
        """
        if num_days > 31:
            warnings.warn('Using a large window of days may give suspect results.', RuntimeWarning)
        if num_days < 3:
            warnings.warn('Using a very small window of days give suspect results.', RuntimeWarning)

        days = pd.unique(self._dates)
        if len(days) <= num_days:
            warnings.warn('Data is smaller than specified window size.', RuntimeWarning)
            for i in range(len(days)):
                yield days[i], days
        else:
            plus_minus = (num_days // 2) + 1
            for i in range(len(days)):
                if i - plus_minus < 0:
                    day_range = days[:num_days]
                elif i + plus_minus >= len(days):
                    day_range = days[len(days) - num_days: len(days)]
                else:
                    day_range = days[i - plus_minus + 1: i + plus_minus]
                yield days[i], day_range

    def calc_property(self, data, fxn, slices=None):
        """Calculate properties of windows.  The property value is set at the central
        point of a window.

        The acceptable functions for calculating properties are:
            self.calc_window_max, self.calc_window_line_length_norm,
            self.calc_window_integral, self.calc_window_line_length,
            self.calc_window_avg, self.calc_window_derivative_std_normed,
            self.calc_window_derivative_avg, self.calc_window_derivative_std,
            self.calc_window_diff_coeff_variation

        Arguments
        ---------
        data: pd.Series
            Time series data of which properties will be calculated.
        fxn: callable
            Function with which properties will be calculated.
        slices, optional: np.array
            Slices for windows.

        Returns
        -------
        ser_vals: pd.Series
            Time series data of calculated property.
        """
        if fxn not in (self.calc_window_max, self.calc_window_line_length_norm,
                       self.calc_window_integral, self.calc_window_line_length,
                       self.calc_window_avg, self.calc_window_derivative_std_normed,
                       self.calc_window_derivative_avg, self.calc_window_derivative_std,
                       self.calc_window_diff_coeff_variation):
            raise ValueError('You have chosen an invalid function.')
        if data is None:
            data = self.data
        if slices is None:
            slices = self.generate_window_slices(data)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        vals = np.apply_along_axis(fxn, 1, data.values[slices])

        ser_vals = pd.Series(np.nan, index=data.index)
        ser_vals.iloc[midpoints] = vals[:]

        return ser_vals

    def calc_spline(self, series, spline_window=None, kind=2):
        """Calculate smoothing spline for a given data set.

        Arguments
        ---------
        series: pd.Series
            Time series data.
        spline_window, optional: int
            Length of time (minutes) in between points with which splines are calculated.
            If None, will use self.window as size.
        kind, optional: int
            Order of spline.

        Returns
        -------
        spline: pd.Series
            Smoothed data from series with same indices as series.
        """
        if spline_window is None:
            spline_window = self.window
        spline_list = []
        for day_str, day_group in series.groupby(series.index.date):
            resampled = day_group.replace([-np.inf, np.inf], np.nan).dropna().\
                        resample(str(spline_window) + 'T').mean()
            xs = np.arange(0, len(resampled))
            ys = resampled.values
            spline = interpolate.interp1d(xs, ys, kind=kind)
            xnew = np.linspace(xs[0], xs[-1], len(day_group))
            y_pred = spline(xnew)
            spline_ser = pd.Series(y_pred)
            spline_ser.index = day_group.index
            spline_ser.replace(np.nan, 0, inplace=True)
            spline_list.append(spline_ser.copy())
        spline = pd.concat(spline_list)
        return spline

    def calc_window_line_length_norm_spline(self, series, slices=None, spline_window=None):
        """Calculate the normalized line length of windows.  Series window line lengths will be
        normalized by the line length of a smoothing spline over that time.

        Arguments
        ---------
        series: pd.Series
            Time series data.
        slices, optional: np.array
            Slices for windows.
        spline_window, optional: int
            Size of step to take in between points used to calculate splines.

        Returns
        -------
        ser_norm_line_length: pd.Series
            Time series of normalized line lenghts for each window.
        """
        if slices is None:
            slices = self.generate_window_slices(self.data)

        spline_ser = self.calc_spline(series, spline_window=spline_window)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        ser_norm_line_length = pd.Series(np.nan, index=series.index)

        for mid, s in zip(midpoints, slices):
            ser_line_length = self.calc_window_line_length(series.values[s])
            spline_line_length = self.calc_window_line_length(spline_ser.values[s])
            ser_norm_line_length.iloc[mid] = ser_line_length / spline_line_length

        return ser_norm_line_length

    def generate_stat_cs(self, num_days=30, model_fxn=np.nanpercentile, percentile=90,
                         smooth_window=0, smooth_fxn=None):
        """Generate a clearsky curve using measured values and model_fxn.  Likely candidates
        for model_fxn would be np.nanmean, np.nanmedian, or np.nanpercentile.  The curve can also be smoothed
        using smooth_window and smooth_fxn.  Smoothing is done using pd.rolling on the series.

        Note: it is generally a good idea to use numpy nan functions as the model function(i.e. np.nanmean, np.nanmedian,
        np.nanpercentile).  The 'normal' functions raise errors if nan values are present in data.

        Arguments
        ---------
        num_days, optional: int
            Size of window (in days) for constructing clearsky curve.
        model_fxn, optional: callable
            Function for determining clearsky curve.
        percentile, optional: float
            Perctile value used in statistical clearsky, ignored if model_fxn is not np.percentile or np.nanpercentile.
        smooth_window, optional: int
            Size (number of data points) for smoothing the statistical clear sky.
        smooth_fxn, optional: callable
            Function for smoothing the clear sky curve.
        """
        stat_cs_list = []

        by_time = self.by_time_of_day_transform()

        for day_to_filter, sample in self.generate_day_range(num_days):
            sample_days = by_time[sample]
            stat_cs = self.stat_cs_per_sample(day_to_filter, sample_days,
                                              model_fxn=model_fxn, percentile=percentile)
            stat_cs_list.append(stat_cs)
        stat_cs_ser = pd.concat(stat_cs_list, axis=0)
        stat_cs_ser.index = self.data.index

        if smooth_window > 0 and smooth_fxn is not None:
            stat_cs_ser = stat_cs_ser.rolling(smooth_window, center=True).apply(smooth_fxn).fillna(0)

        return stat_cs_ser

    def stat_cs_per_sample(self, day_to_filter, sample_days, model_fxn=np.nanpercentile, percentile=90):
        """Filter measurements by time of day based on deviation from fxn.

        Arguments
        ---------
        day_to_filter: datetime.date
            day which will be filtered
        sample_days: pd.DataFrame
            DataFrame of measured values.  Rows are time of day and columns are dates.
        model_fxn, optional: callable
            Function that will be used to construct the statistical clearsky curve.
        percentile, optional: float
            Percentile value for clearsky curve construction if percentile based function used.
        """
        # fixing indices is important - the by_time_of_day_transform will fill daylight savings/etc
        # which confuses indexing central vals
        correct_indices = self.data.loc[self._dates == day_to_filter].index
        correct_times = correct_indices.time
        sample_days = sample_days[sample_days.index.isin(correct_times)]

        if model_fxn in (np.percentile, np.nanpercentile):
            args = ([percentile])
        else:
            args = ()

        central_vals = pd.Series(sample_days.replace(0, np.nan).apply(model_fxn, axis=1, args=args),
                                 name='central').fillna(0)
        central_vals.index = correct_indices

        return central_vals

    def deviation_time_filter(self, num_days=30, model_fxn=np.nanmean,
                              mode='relative', percentile=90, dev_range=(.8, 1.2), verbose=False):
        """Filter measurements by time of day based on deviation a statistical clearsky curve.

        Note: it is generally a good idea to use numpy nan functions (i.e. np.nanmean, np.nanmedian,
        np.nanpercentile).  The 'normal' functions will raise errors if nan values are present in data.

        Arguments
        ---------
        num_days, optional: int
            Size of window (in days) for building statistical clearsky.
        model_fxn, optional: callable
            Function for calculating central values.
        mode, optional: str
            relative:
                Deviation calculated as ratio of measured to central value.
            direct:
                Deviation is difference of measured and central value.
            zscore:
                Zscore calculated for each time period based on window.
        dev_range, optional: tuple(float, float)
            Range of allowed deviation.  First value is lower limit, second value is upper limit.

        """
        if mode in ('direct', 'relative'):
            stat_cs = self.generate_stat_cs(num_days=num_days, model_fxn=model_fxn, percentile=percentile)
            if mode == 'direct':
                dev = self.data - stat_cs
            elif mode == 'relative':
                dev = self.data / stat_cs
        elif mode == 'zscore':
            means = self.generate_stat_cs(num_days=num_days, model_fxn=np.nanmean)
            stds = self.generate_stat_cs(num_days=num_days, model_fxn=np.nanstd)
            dev = (self.data - means) / stds
        else:
            raise ValueError("mode must be 'direct', 'relative', or 'zscore'.")

        mask = pd.Series((dev >= dev_range[0]) & (dev <= dev_range[1]), index=self.data.index)
        self.filter_mask = mask

    def standard_detection(self, verbose=False, splines=False, spline_window=None, metric_tol=.01):
        """Determine clear sky periods based on irradiance measurements.  Central value of window
        is labeled clear if window passes test.

        Arguments
        ---------
        splines, optional: bool
            Use splines to calculate normalized line length.
        spline_window, optional:
            Frequency of time periods for calculating splines.
        metric_tol, optional: float
            tolerance for determining clear skies
        verbose: bool
            Whether or not to return components used to determine is_clear.

        Returns
        -------
        is_clear: pd.Series
            boolean time series of clear times
        components, optional: dict
            contains series of normalized lengths, local integrals, and calculated metric
        """
        is_clear = pd.Series(False, self.data.index)

        components = self.calc_components(splines=splines, spline_window=spline_window)

        is_clear = ((components['metric'] <= metric_tol) & (self.data > 0.0) & (self.filter_mask))

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def calc_components(self, splines=False, spline_window=None, slices=None):
        """Calculate normalized distances and integrals of moving window.  Values
        are reported at the central index of the window.

        Arguments
        ---------
        slices: np.ndarray
            slices for windows

        Returns
        -------
        result: pd.DataFrame
            Contains calculated properties for clearsky detection.
        """
        if slices is None:
            slices = self.generate_window_slices(self.data)

        local_integrals = self.calc_property(self.data, self.calc_window_avg)
        local_integrals.name = 'local_integrals'
        if splines:
            local_distances = \
                    self.calc_window_line_length_norm_spline(self.data, slices=slices, spline_window=spline_window)
        else:
            local_distances = \
                    self.calc_property(self.data, self.calc_window_line_length_norm)
        local_distances.name = 'local_distances'

        metric = self.calc_cloudiness_metric(local_distances.values,
                                             local_integrals.values)
        metric = pd.Series(metric, index=self.data.index, name='metric')

        result = pd.concat([local_distances, local_integrals, metric], axis=1)

        return result

    def calc_cloudiness_metric(self, distances, integrals):
        """Calculate the cloudiness metric.

        Cloudiness = log(distances) / log(integrals)

        Arguments
        ---------
        distances: np.array
            Normalized distsances of windows.
        integrals: np.array
            Local integral of irradiance of a window.

        Returns
        -------
        metric: np.array
            Metric values.

        """
        # metric = np.log(distances) / np.log(integrals)
        # metric = np.log(distances) / np.log(integrals)
        metric = distances**2 / integrals
        return metric

    def mean_detection(self, verbose=False, splines=False, spline_window=None, metric_tol=.01):
        """Determine clear sky periods based on irradiance measurements.  Central value
        of window is labeled clear if the average value of the window is at or below metric_tol.
        The cloudiness metric is smoothed by averaging the points +/- (self.window / 2).

        Arguments
        ---------
        splines, optional: bool
            Use splines to calculate normalized line length.
        spline_window, optional:
            Frequency of time periods for calculating splines.
        metric_tol, optional: float
            tolerance for determining clear skies
        verbose: bool
            Whether or not to return components used to determine is_clear.

        Returns
        -------
        is_clear: pd.Series
            Boolean time series of clear times.
        components, optional: pd.DataFrame
            Contains series of normalized lengths, local integrals, and calculated metric.
        """
        is_clear = pd.Series(False, self.data.index)

        components = self.calc_components(splines=splines, spline_window=spline_window)

        slices = self.generate_window_slices(self.data)

        components = self.calc_components(slices=slices)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        means = np.apply_along_axis(self.calc_window_avg, 1, components['metric'].values[slices], weights=None)

        is_clear.iloc[midpoints] = (means <= metric_tol) & (self.data[midpoints] > 0.0) & (self.filter_mask.iloc[midpoints])

        if verbose:
            return is_clear, components
        else:
            return is_clear

    def democratic_detection(self, vote_pct=.75, splines=False, spline_window=None, verbose=False, metric_tol=.01):
        """Determine clear sky periods based on irradiance measurements.

        Central value
        of window is labeled clear if the average value of the window is at or below metric_tol.
        The clarity of a given point is determined by calculating the percent of windows it is in
        that are <= metric_tol.  Each point is in self.window windows (right-most point all the
        way to left-most point).

        Arguments
        ---------
        splines, optional: bool
            Use splines to calculate normalized line length.
        spline_window, optional:
            Frequency of time periods for calculating splines.
        metric_tol, optional: float
            tolerance for determining clear skies
        verbose: bool
            Whether or not to return components used to determine is_clear.

        Returns
        -------
        is_clear: pd.Series
            Boolean time series of clear times.
        components, optional: pd.DataFrame
            Contains series of normalized lengths, local integrals, and calculated metric.
        """
        is_clear = pd.Series(False, self.data.index)

        components = self.calc_components(splines=splines, spline_window=spline_window)

        slices = self.generate_window_slices(self.data)

        midpoints = np.apply_along_axis(self.get_midval, 1, slices)
        pcts = np.apply_along_axis(self.calc_pct, 1, components['metric'].values[slices], metric_tol)

        is_clear.iloc[midpoints] = (pcts >= vote_pct) & (self.data[midpoints] > 0.0) & (self.filter_mask.iloc[midpoints])

        if verbose:
            return is_clear, components
        else:
            return is_clear


if __name__ == '__main__':
    main()
