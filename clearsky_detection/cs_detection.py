
import warnings
import os

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
import pvlib
import pytz

class ClearskyDetection(object):
    """Class for detecting clearsky based on NSRDB data."""

    def __init__(self, df, copy=True):
        """Initialize members.

        Arguments
        ---------
        nsrdb_df: pd.DataFrame
            NSRDB data.
        ground_df: pd.DataFrame
            Ground based data.
        """
        if copy:
            self.df = df.copy()
        else:
            self.df = df
        # if str(self.df.index.tz) != 'UTC':
        #     self.df.index = self.df.index.tz_convert('UTC')

    @classmethod
    def read_nsrdb_dir(cls, dir_path, timezone, keepers=['GHI', 'Clearsky GHI', 'Cloud Type'], file_ext='csv'):
        """Read directory of NSRDB files.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

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
        """
        if file_ext.lower() not in ('csv'):
            raise NotImplementedError('Only accept CSV files at this time.')
        files = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith(file_ext)]
        df = pd.concat([pd.read_csv(f, header=2) for f in files])
        df.index = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
        tmp_list = []
        for index in df.index:
            try:
                index.tz_localize(timezone)
                tmp_list.append(index)
            except:
                pass
        df = df[df.index.isin(tmp_list)]
        df.index = df.index.tz_localize(timezone)
        # df.index = df.index.tz_localize('UTC')
        # df.index = df.index.tz_convert(timezone)
        df = df.sort_index()
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='30min')).fillna(0)
        df = df[~df.index.duplicated(keep='first')]
        df.index = df.index.tz_convert('UTC')
        return cls(df[keepers])

    @classmethod
    def read_snl_rtc(cls, file_w_path, timezone, keepers=['GHI']):
    # def read_snl_rtc(cls, file_w_path, timezone1, timezone2, keepers=['GHI']):
        """Read SNL RTC data into file.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

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
        """
        df = pd.read_csv(file_w_path, parse_dates=['Date-Time'], index_col=['Date-Time'])
        tmp_list = []
        for index in df.index:
            try:
                index.tz_localize(timezone)
                tmp_list.append(index)
            except:
                pass
        df = df[df.index.isin(tmp_list)]
        df.index = df.index.tz_localize(timezone)
        df = df.sort_index()
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')).fillna(0)
        df = df[~df.index.duplicated(keep='first')]
        df['GHI'] = df['Global_Wm2']
        df.index = df.index.tz_convert('UTC')
        return cls(df[keepers])

    @classmethod
    def read_srrl_dir(cls, dir_path, timezone, file_ext='txt', keepers=['GHI']):
        """Read directory of SRRL files into a dataframe.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

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
        """
        files = [os.path.join(dir_path, i) for i in os.listdir(dir_path) if i.endswith('txt')]
        df = pd.concat([pd.read_csv(f) for f in files])
        df.index = pd.to_datetime(df['DATE (MM/DD/YYYY)'] + ' ' + df['MST'], format='%m/%d/%Y %H:%M')
        tmp_list = []
        for index in df.index:
            try:
                index.tz_localize(timezone)
                tmp_list.append(index)
            except:
                pass
        df = df[df.index.isin(tmp_list)]
        df.index = df.index.tz_localize(timezone)
        # df.index = df.index.tz_convert('UTC')
        #df.index = df.index.tz_convert('US/Mountain')
        df = df.sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')).fillna(0)
        df['GHI'] = df['Global PSP [W/m^2]']
        try:
            df['GHI'].fillna(df['Global PSP (cor) [W/m^2]'], inplace=True)
        except:
            df['GHI'] = df['GHI'].fillna(0)
        df.index = df.index.tz_convert('UTC')
        return cls(df[keepers])

    @classmethod
    def read_pickle(cls, filename, compression=None):
        """Read dataframe from pickle file.

        Arguments
        ---------
        filename: str
            Name of pickle file.
        """
        df = pd.read_pickle(filename)
        return cls(df)

    # def __trim_bad_indices_tz(self, indices, timezone):
    #     tmp_list = []
    #     for index in indices:
    #         try:
    #             index.tz_localize(timezone)
    #             tmp_list.append(index)
    #         except:
    #             pass
    #     return tmp_list

    def to_pickle(self, filename, overwrite=False, compression=None):
        """Dump dataframe to pickle file.

        Arguments
        ---------
        filename: str
            Name of pickle file.
        overwrite, optional: bool
            Overwrite file if it exists.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError('The specified file already exists.  Change name or set overwrite.')
        self.df.to_pickle(filename)

    def robust_rolling_smooth(self, col, window, label=None, overwrite=False):
        """Smooth column of data robustly (median + mean).

        Arguments
        ---------
        col: str
            Column in dataframe.
        window: int
            Number of points to include in window.
        label, optional: str
            Name of smoothed column.
        overwrite, optional: bool
            Overwrite new column or not.
        """
        if label is None:
            label = col + ' smooth'
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Provide label name or allow for overwrite.')
        self.df[label] = \
            self.df[col].rolling(window, center=True).median().fillna(0).rolling(window, center=True).mean().fillna(0)

    def trim_dates(self, x0=None, x1=None):
        """Trim date range of dataframe.

        The x0 is inclusive, x1 is exclusive.  If either are None, assume no trimming in said direction.

        Arguments
        ---------
        x0, optional: datetime or str
            Earlier date (inclusive).
        x1, optional: datetime or str
            Later date (exclusive).
        """
        if x0 is not None:
            self.df = self.df[self.df.index >= x0]
        if x1 is not None:
            self.df = self.df[self.df.index < x1]

    def intersection(self, other_indices):
        """Modify indices such that dataframe intersects with different set of indices.

        Arguments
        ---------
        other_indices: pd.DateTimeIndex
            Indices to intersect with.
        """
        indices = self.df.index.intersection(other_indices)
        self.df = self.df[self.df.index.isin(indices)]

    def by_time_of_day_transform(self, data='GHI'):
        """Transform time series (linear) to matrix representation.

        The dataframe will be organized as Date (columnss) X Timestample (rows).

        Arguments
        ---------
        data, optional: str
            Which data column (nsrdb/ground).  Assumed to be 'GHI'.

        Returns
        -------
        by_time: pd.DataFrame
            DataFrame with time of day as rows and dates as columns.
        """
        by_time = pd.pivot_table(self.df, index=self.df.index.time, columns=self.df.index.date, values=data)
        return by_time

    def generate_statistical_clearsky(self, data='GHI', num_days=30,
                                      model_fxn=np.nanpercentile,
                                      percentile=90, smooth_window=0, smooth_fxn=None,
                                      label='Clearsky GHI stat', overwrite=False):
        """Generate statistical clearsky for either dataset.

        Arguments
        ---------
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
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        # old_tz = self.df.index.tz
        # old_idx = self.df.index
        # self.df.index = self.df.index.tz_convert('UTC')
        by_time = self.by_time_of_day_transform(data)
        dates = self.df.index.date

        stat_cs_list = []
        for day_to_filter, sample in self.generate_day_range(dates, num_days):
            sample_days = by_time[sample]
            stat_cs = self.stat_cs_per_sample(dates, day_to_filter, sample_days,
                                              model_fxn, percentile)
            stat_cs_list.append(stat_cs)

        stat_cs_ser = pd.concat(stat_cs_list, axis=0)
        stat_cs_ser.index = self.df.index

        if smooth_window > 0 and smooth_fxn is not None:
            stat_cs_ser = stat_cs_ser.rolling(smooth_window, center=True).\
                apply(smooth_fxn).fillna(0)

        self.df[label] = stat_cs_ser
        # self.df.index = self.df.index.tz_convert(old_tz)
        # self.df.index = old_idx

    def stat_cs_per_sample(self, dates, day_to_filter, sample_days,
                           model_fxn=np.nanpercentile, percentile=90):
        """Filter measurements by time of day based on deviation from fxn.

        Arguments
        ---------
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
        correct_indices = self.df.loc[dates == day_to_filter].index
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

    def generate_pvlib_clearsky(self, latitude, longitude, altitude=None, label='Clearsky GHI pvlib', overwrite=False):
        """Generate pvlib clearsky curve.

        Arguments
        ---------
        lat: float
            Latitutde.
        lon: float
            Longitude.
        elevation, optional: float
            Elevation of system (this is not required by Location object).
        label: str
            Label of clearsky data points.
        overwrite: bool
            Allow column to be overwritten if exists.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        loc = pvlib.location.Location(latitude, longitude, altitude=altitude)
        clear_skies = loc.get_clearsky(self.df.index)
        clear_skies = pd.Series(clear_skies['ghi'])
        clear_skies.index = self.df.index
        self.df[label] = clear_skies

    def calc_distance_to_solar_noon(self, model_col, label='time_to_solarnoon', overwrite=False):
        """Get solar position based on indices and optionally calculate distance to solar noon for each part of the day.

        There is probably a faster way to do this.

        Arguments
        ---------
        model_col: str
            Column of model GHI.
        label, optional: str
            Name of calculated time to solar noon.
        overwrite, optional: bool
            Overwrite label if it exists.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists. Set overwrite to true or set label name.')
        max_indices = []
        for _, group in self.df[model_col].groupby(self.df[model_col].index.date):
            max_indices.append(group.idxmax())
        max_indices = pd.DataFrame(max_indices)
        max_indices.index = max_indices.values
        res_list = []
        # for idx in max_indices.index:
        #     res = self.df.iloc[self.df.index.get_loc(idx, method='nearest')]
        #     res_list.append(res)
        # return res_list
        for idx in self.df.index:
            res = max_indices.iloc[max_indices.index.get_loc(idx, method='nearest')]
            res_list.append(res)
        nearest_max = pd.DataFrame(res_list, columns=['nearest_max'])
        self.df[label] = np.abs(self.df.index - nearest_max['nearest_max'].values).total_seconds() / 60
        # return res_list


    #
    #
    # #     # dist_list = []
    # #     # for idx in self.df.index:
    # #     #     # subset = [i for i in max_indices if i in pd.date_range(start=idx - pd.Timedelta('1 days'), end=idx + pd.Timedelta('1 days'), freq=self.df.index.freq)]
    # #     #     date_range = [(idx - pd.Timedelta('1 days')).date(), (idx + pd.Timedelta('1 days')).date(), idx.date()]
    # #     #     subset = np.asarray([i for i in max_indices if i.date() in date_range])
    # #     #     distances = [np.abs((idx - i).total_seconds() / 60) for i in subset]
    # #     #     dist_list.append(np.min(distances))

    def pvlib_clearsky_detect(self, scale=False, measured='GHI',
                              modeled='Clearsky GHI pvlib', label='sky_status pvlib', window=10, overwrite=False):
        """Detect clearsky using PVLib methods.

        Arguments
        ---------
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
        overwrite, optional: bool
            Allow column to be overwritten if label exists.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        alpha = 1.0
        if scale:
            status, _, alpha = \
                pvlib.clearsky.detect_clearsky(self.df[measured], self.df[modeled],
                                               self.df.index, window, return_components=True)
        else:
            status = pvlib.clearsky.detect_clearsky(self.df[measured], self.df[modeled], self.df.index, window)
        self.df[label] = status
        if scale:
            self.df[modeled + ' scaled'] = alpha * self.df[modeled]

    def set_nsrdb_sky_status(self, label='sky_status', overwrite=False):
        """Set sky status target for machine learning applications for nsrdb data.

        Arguments
        ---------
        label, optional: str
            Column name for new sky status.
        overwrite, optional: bool
            Allow column to be overwritten in column exists
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        self.df[label] = (self.df['Cloud Type'] == 0) & (self.df['GHI'] > 0)

    def calc_abs_ratio_diff(self, col1, col2, label='abs_diff_ratio', overwrite=False):
        """Absolute difference of the ratio of two series from 1.

        Arguments
        ---------
        col1: str
            Column for numerator.
        col2: str
            Column for denominator.
        label: str
            Name of resultant column.
        overwrite, optional: bool
            Overwite column if it exists.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        ratio = (self.df[col1] / self.df[col2]).replace([-np.inf, np.inf, np.nan], 1)
        self.df[label] = np.abs(1 - ratio)

    def calc_ratio(self, col1, col2, label='ratio', overwrite=False):
        """Ratio of two columns of dataframe.

        Arguments
        ---------
        col1: str
            Column for numerator.
        col2: str
            Column for denominator.
        label: str
            Name of resultant column.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        ratio = (self.df[col1] / self.df[col2]).replace([-np.inf, np.inf, np.nan], 1)  # replace with 1 for 0/0 case
        self.df[label] = ratio

    def calc_window_stat(self, window, col, label, overwrite=False):
        """Calculates window-based statistics on columns.

        Arguments
        ---------
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
        overwrite, optional: bool
            Allow columns to be overwritten if they exist.
        """
        test_labels = [label + i for i in [' mean', ' std', ' max', ' min', ' range']]
        if any(i in self.df.keys() for i in test_labels) and not overwrite:
            raise RuntimeError('A label already exists.  Use new label name set overwrite to True.')
        self.df[label + ' mean'] = self.df[col].rolling(window, center=True).mean().fillna(0)
        self.df[label + ' std']  = self.df[col].rolling(window, center=True).std().fillna(0)
        self.df[label + ' max']  = self.df[col].rolling(window, center=True).max().fillna(0)
        self.df[label + ' min']  = self.df[col].rolling(window, center=True).min().fillna(0)
        self.df[label + ' range']  = self.df[label + ' max'] - self.df[label + ' min']

    def calc_all_window_metrics(self, window, dx, col1='GHI', col2='Clearsky GHI',
                                ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=False):
        """Setup dataframes for machine learning.

        Arguments
        ---------
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
        overwrite, optional: bool
            Allow column to be overwritten.
        """
        if (ratio_label in self.df.keys() or abs_ratio_diff_label in self.df.keys()) and not overwrite:
            raise RuntimeError('A label already exists.  Set overwrite to True or pick new label name.')
        self.calc_ratio(col1, col2, ratio_label, overwrite=overwrite)
        self.calc_abs_ratio_diff(col1, col2, abs_ratio_diff_label, overwrite=overwrite)

        self.calc_window_stat(window, col1, col1, overwrite=overwrite)
        self.calc_window_stat(window, col2, col2, overwrite=overwrite)

        self.calc_window_stat(window, ratio_label, ratio_label, overwrite=overwrite)
        self.calc_window_stat(window, abs_ratio_diff_label, abs_ratio_diff_label, overwrite=overwrite)

        label = col1 + ' gradient'
        self.df[label] = np.gradient(self.df[col2], dx)
        self.calc_window_stat(window, label, label, overwrite=overwrite)

        label = col2 + ' gradient'
        self.df[label] = np.gradient(self.df[col2], dx)
        self.calc_window_stat(window, label, label, overwrite=overwrite)

        label = ratio_label + ' gradient'
        self.df[label] = np.abs(np.gradient(self.df[ratio_label], dx))
        self.calc_window_stat(window, label, label, overwrite=overwrite)

        label = abs_ratio_diff_label + ' gradient'
        self.df[label] = np.abs(np.gradient(self.df[abs_ratio_diff_label], dx))
        self.calc_window_stat(window, label, label, overwrite=overwrite)

    def fit_model(self, feature_cols, target_cols, clf,
                  conf_matrix=True, cv=True):
        """Fit machine learning model based on data frame.

        Arguments
        ---------
        feature_cols: list-like
            Column names to use as features in ML model.
        target_cols: list-like
            Column names to use as target label.
        clf: sklearn estimator
            Object with fit and predict methods.
        train_test_split, optional: bool
            Perform train_test_split.
        cv, optional: bool
            Perform cross validation.
        """
        X = self.df[feature_cols].values
        y = self.df[target_cols].values.flatten()
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
        if cv:
            scores = model_selection.cross_val_score(clf, X_train, y_train)
            mean_score = np.round(np.mean(scores), 4)
            std_score = np.round(np.std(scores), 4)
            print('CV score: {} +/- {}'.format(mean_score, std_score))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred)
        print('Train/test split score: {}'.format(score))
        return clf
