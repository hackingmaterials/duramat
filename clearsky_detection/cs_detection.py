
import warnings

import numpy as np
import pandas as pd
import pvlib

class ClearskyDetection(object):
    """Class for detecting clearsky based on NSRDB data."""

    def __init__(self, nsrdb_df, ground_df, copy=True):
        """Initialize members.

        Arguments
        ---------
        nsrdb_df: pd.DataFrame
            NSRDB data.
        ground_df: pd.DataFrame
            Ground based data.
        """
        if 'GHI' not in nsrdb_df.keys() or 'Clearsky GHI' not in nsrdb_df.keys():
            raise KeyError('NSRDB data must have GHI and Clearsky GHI columns.')
        if 'GHI' not in ground_df.keys():
            raise KeyError('Ground data must have GHI column.')
        if copy:
            self.nsrdb_df = nsrdb_df.copy()
            self.ground_df = ground_df.copy()
        else:
            self.nsrdb_df = nsrdb_df
            self.ground_df = ground_df
        self.nsrdb_df.index = self.nsrdb_df.index.tz_convert('UTC')
        self.ground_df.index = self.ground_df.index.tz_convert('UTC')
        self.dfs = {'nsrdb': self.nsrdb_df, 'ground': self.ground_df}

    def by_time_of_day_transform(self, which, data='GHI'):
        """Transform time series (linear) to matrix representation.

        The dataframe will be organized as Date (columnss) X Timestample (rows).

        Arguments
        ---------
        which: str
            Which data set (nsrdb/ground).
        data, optional: str
            Which data column (nsrdb/ground).  Assumed to be 'GHI'.

        Returns
        -------
        by_time: pd.DataFrame
            DataFrame with time of day as rows and dates as columns.
        """
        tmp = self.dfs[which]
        # tmp = pd.DataFrame(self.data, index=self.data.index)
        by_time = pd.pivot_table(tmp, index=tmp.index.time, columns=tmp.index.date, values=data)
        return by_time

    def generate_statistical_clearsky(self, which, data='GHI', num_days=30,
                                      model_fxn=np.nanpercentile,
                                      percentile=90, smooth_window=0, smooth_fxn=None,
                                      result='Clearsky GHI stat'):
        """Generate statistical clearsky for either dataset.

        Arguments
        ---------
        which: str
            Either 'nsrdb' or 'ground'.
        data, optional: str
            Data column to generate clearsky, assumed to be 'GHI'.
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
        by_time = self.by_time_of_day_transform(which, data)
        dates = self.dfs[which].index.date

        stat_cs_list = []
        for day_to_filter, sample in self.generate_day_range(dates, num_days):
            sample_days = by_time[sample]
            stat_cs = self.stat_cs_per_sample(which, dates,
                                              day_to_filter, sample_days,
                                              model_fxn, percentile)
            stat_cs_list.append(stat_cs)

        stat_cs_ser = pd.concat(stat_cs_list, axis=0)
        stat_cs_ser.index = self.dfs[which].index

        if smooth_window > 0 and smooth_fxn is not None:
            stat_cs_ser = stat_cs_ser.rolling(smooth_window, center=True).\
                apply(smooth_fxn).fillna(0)

        self.dfs[which][result] = stat_cs_ser

    def stat_cs_per_sample(self, which, dates, day_to_filter, sample_days,
                           model_fxn=np.nanpercentile, percentile=90):
        """Filter measurements by time of day based on deviation from fxn.

        Arguments
        ---------
        which: str
            Either 'nsrdb' or 'ground'.
        dates: list-like
            Unique dates in data.
        day_to_filter: datetime.date
            Day which will be filtered.
        sample_days: pd.DataFrame
            DataFrame of measured values (rows are time of day and columns are dates).
        model_fxn, optional: callable
            Function that will be used to construct the statistical clearsky curve.
        percentile, optional: float
            Percentile value for clearsky curve construction if percentile based function used.
        """
        # fixing indices is important - the by_time_of_day_transform will fill daylight savings/etc
        # which confuses indexing central vals
        correct_indices = self.dfs[which].loc[dates == day_to_filter].index
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

    def generate_day_range(self, dates, num_days=30):
        """Generates groups of days for statistical analysis.

        Arguments
        ---------
        window_size, optional: int
            Size of window (in days).
        dates: list-like
            Unique dates in data.

        Yields
        ------
        day_range: tuple
            (Day of interest, date of days (+/- window_size / 2))
        """
        if num_days > 31:
            warnings.warn('Using a large window of days may give suspect results.', RuntimeWarning)
        if num_days < 3:
            warnings.warn('Using a very small window of days give suspect results.', RuntimeWarning)

        days = pd.unique(dates)
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

    def generate_pvlib_clearsky(self, which, tilt, elevation, azimuth, lat, lon, result='Clearsky GHI'):
        """Generate pvlib clearsky curve.

        Arguments
        ---------
        which: str
            Dataset to genearate curve for.
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
        """
        sys_no_loc = pvlib.pvsystem.PVSystem(surface_tilt=tilt, surface_azimuth=azimuth)
        sys_loc = pvlib.location.Location(lat, lon, altitude=elevation)
        sys = pvlib.pvsystem.LocalizedPVSystem(pvsystem=sys_no_loc, location=sys_loc)
        clear_skies = sys.get_clearsky(self.dfs[which].index)
        clear_skies = pd.Series(clear_skies['ghi'])
        clear_skies.index = self.dfs[which].index
        self.dfs[which][result] = clear_skies

    def pvlib_clearsky_detect(self, which, scale=False, measured='GHI',
                              modeled='Clearsky GHI', result='sky_status', window=10):
        """Detect clearsky using PVLib methods.

        Arguments
        ---------
        which: str
            Dataset to detect clear peridos for.
        scale, optional: bool
            Scale modeled column by factor determined by PVLib.
        measured, optional: str
            Column to detect.
        modeled, optional: str
            Column of modeled clearsky (reference for measured).
        result, optional: str
            Column name of results.
        window, optional: int
            Size of window for PVLib detection.
        """
        df = self.dfs[which]
        alpha = 1.0
        if scale:
            status, _, alpha = \
                pvlib.clearsky.detect_clearsky(df[measured], df[modeled], window, df.index, return_components=True)
        else:
            status = pvlib.clearsky.detect_clearsky(df[measured], df[modeled], window, df.index)
        df[result] = status
        df[modeled] = alpha * df[modeled]

    def calc_abs_ratio_diff(self, which, col1, col2, label):
        """Absolute difference of the ratio of two series from 1.

        Arguments
        ---------
        which: str
            Which data set to operate on.
        col1: str
            Column for numerator.
        col2: str
            Column for denominator.
        label: str
            Name of resultant column.
        """
        df = self.dfs[which]
        ratio = (df[col1] / df[col2]).replace([-np.inf, np.inf, np.nan], 1)
        df[label] = np.abs(1 - ratio)

    def calc_ratio(self, which, col1, col2, label):
        """Ratio of two columns of dataframe.

        Arguments
        ---------
        which: str
            Which data set to operate on.
        col1: str
            Column for numerator.
        col2: str
            Column for denominator.
        label: str
            Name of resultant column.
        """
        df = self.dfs[which]
        ratio = (df[col1] / df[col2]).replace([-np.inf, np.inf, np.nan], 1)
        df[label] = ratio

    def calc_window_stat(self, which, window, col, label):
        """Calculates window-based statistics on columns.

        Arguments
        ---------
        which: str
            Which data set to operate on.
        window: int
            Size of window for calculating metrics.  Defined as the number of timesteps to use in the window.
            For example, if data is 30 min frequency and window=3, the window will span 1 hour.
        col: str
            Which column to use.
        label: str
            Result column label.
        apply_fxn, optional: callable
            Function to apply to columns before computing statistics.
        apply_fxn_args, optional: list-like
            Options for apply_fxn.
        """
        df = self.dfs[which]
        df[label + ' mean'] = df[col].rolling(window, center=True).mean().fillna(0)
        df[label + ' std']  = df[col].rolling(window, center=True).std().fillna(0)
        df[label + ' max']  = df[col].rolling(window, center=True).max().fillna(0)
        df[label + ' min']  = df[col].rolling(window, center=True).min().fillna(0)
        df[label + ' range']  = df[label + ' max'] - df[label + ' min']

    def calc_all_window_metrics(self, which, window, dx, col1='GHI', col2='Clearsky GHI',
                                ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio'):
        """Setup dataframes for machine learning.

        Arguments
        ---------
        which: str
            Which data set to operate on.
        window: int
            Size of window for statistics calculations.
        dx: float
            Step size for gradient calculation.
        col1: str
            Numerator for ratio and abs_diff_ratio calculation.
        col2: str
            Denominator for ratio and abs_diff_ratio calculation.
        ratio_label: str
            Label for ratio columns.
        abs_ratio_diff_label: str
            Label for absolute difference ratio columns.
        """
        self.calc_ratio(which, col1, col2, ratio_label)
        self.calc_abs_ratio_diff(which, col1, col2, abs_ratio_diff_label)

        self.calc_window_stat(which, window, col1, col1)
        self.calc_window_stat(which, window, col2, col2)

        self.calc_window_stat(which, window, ratio_label, ratio_label)
        self.calc_window_stat(which, window, abs_ratio_diff_label, abs_ratio_diff_label)

        label = col1 + ' gradient'
        self.dfs[which][label] = np.gradient(self.dfs[which][col2], dx)
        self.calc_window_stat(which, window, label, label)

        label = col2 + ' gradient'
        self.dfs[which][label] = np.gradient(self.dfs[which][col2], dx)
        self.calc_window_stat(which, window, label, label)

        label = ratio_label + ' gradient'
        self.dfs[which][label] = np.gradient(self.dfs[which][ratio_label], dx)
        self.calc_window_stat(which, window, label, label)

        label = abs_ratio_diff_label + ' gradient'
        self.dfs[which][label] = np.gradient(self.dfs[which][abs_ratio_diff_label], dx)
        self.calc_window_stat(which, window, label, label)
