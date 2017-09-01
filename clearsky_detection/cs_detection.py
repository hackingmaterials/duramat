
import warnings
import os
import concurrent.futures as cf
import multiprocessing as mp

import numpy as np
from scipy import linalg
from scipy import optimize
import pandas as pd
import pvlib
import utils

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
    def read_ornl_file(cls, filename, timezone='EST', keepers = ['GHI']):
        df = pd.read_csv(filename)
        df.index = pd.to_datetime(df['DATE (MM/DD/YYYY)'] + ' ' + df['EST'], format='%m/%d/%Y %H:%M')
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
        df = df[~df.index.duplicated(keep='first')]
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='1min')).fillna(0)
        df['GHI'] = df['Global Horizontal [W/m^2]']
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

    def scale_model(self, meas_col, model_col, status_col=None):
        """Scale model values to measured values based on clear sky periods.

        Arguments
        ---------
        meas_col: str
            Column of measured GHI values.
        model_col: str
            Column of model GHI values.
        status_col: str
            Column of clear/cloudy classification (must be binary).
        """
        if status_col is not None:
            clear_meas = self.df[self.df[status_col]][meas_col]
            clear_model = self.df[self.df[status_col]][model_col]
        else:
            clear_meas = self.df[meas_col]
            clear_model = self.df[model_col]
        alpha = 1
        def rmse(alpha):
            sqr_err = (clear_meas - (alpha * clear_model))**2
            return np.sqrt(np.mean(sqr_err))
        alpha = optimize.minimize_scalar(rmse).x
        self.df[model_col] = alpha * self.df[model_col]

    # def iter_predict(self, feature_cols, meas_col, model_col, clf, window, n_iter=20, tol=1.0e-8, ml_label='sky_status iter', smooth=False, overwrite=True):
    #     """Predict clarity based using classifier that iteratively fits the model column
    #     to the measured column based on clear points.
    #
    #     This function WILL overwrite columns that already exist in the data frame.
    #
    #     Arguments
    #     ---------
    #     feature_cols: list-like
    #         Column names to use as features in ML model.
    #     clf: sklearn estimator
    #         Object with fit and predict methods.
    #     meas_col: str
    #         Column of measured data.
    #     model_col: str
    #         Column of model data.
    #     n_iter, optional: int
    #         Number of iterations for fitting model to measured column.
    #     tol, optoinal: float
    #         Criterion for convergence of modeled and measured clear points.
    #     ml_label, optional: str
    #         Label for predicted clear/cloudy points.
    #     smooth, optional: bool
    #         Smooth results.  Smoothing is aggressive as a point must be clear
    #         in every window it appears in.
    #     overwrite, optional: bool
    #         Permission to overwrite columns if they exist.
    #     """
    #     alpha = 1
    #     for it in range(n_iter):
    #         print(it + 1, alpha)
    #         self.calc_all_window_metrics(window, col1=meas_col, col2=model_col,
    #                                      ratio_label='ratio', abs_ratio_diff_label='abs_diff_ratio', overwrite=overwrite)
    #         X = self.df[feature_cols].values
    #         y_pred = clf.predict(X)
    #         clear_meas = self.df[y_pred][meas_col]
    #         clear_model = self.df[y_pred][model_col]
    #         alpha_last = alpha
    #         def rmse(alpha):
    #             return np.sqrt(np.mean((clear_meas - (alpha * clear_model))**2))
    #         min_scalar = optimize.minimize_scalar(rmse)
    #         alpha = min_scalar.x
    #         if np.abs(alpha - alpha_last) < tol:
    #             break
    #         self.df[model_col] = self.df[model_col] * alpha
    #     if it == n_iter - 1:
    #         warnings.warn('Scaling did not converge.', RuntimeWarning)
    #     self.df[ml_label] = y_pred
    #     if smooth:
    #         self.smooth_ml_label(window * 2, ml_label)
    #     return self.df[ml_label]

    # def iter_predict_daily(self, feature_cols, meas_col, model_col, clf, window, n_iter=20,
    #                              tol=1.0e-8, ml_label='sky_status iter', smooth=False, overwrite=True):
    #     """Predict clarity based using classifier that iteratively fits the model column
    #     to the measured column based on clear points.
    #
    #     This method differs from iter_predict method because it predicts/scales on individual days, not the entire data set.
    #     This function WILL overwrite columns that already exist in the data frame.
    #
    #     Method should be refactored.  Calculating features and scaling model GHI is done across entire data set even though we only need one day at a time.
    #
    #     Arguments
    #     ---------
    #     feature_cols: list-like
    #         Column names to use as features in ML model.
    #     clf: sklearn estimator
    #         Object with fit and predict methods.
    #     meas_col: str
    #         Column of measured data.
    #     model_col: str
    #         Column of model data.
    #     n_iter, optional: int
    #         Number of iterations for fitting model to measured column.
    #     tol, optoinal: float
    #         Criterion for convergence of modeled and measured clear points.
    #     ml_label, optional: str
    #         Label for predicted clear/cloudy points.
    #     smooth, optional: bool
    #         Smooth results.  Smoothing is aggressive as a point must be clear
    #         in every window it appears in.
    #     overwrite, optional: bool
    #         Permission to overwrite columns if they exist.
    #     """
    #     # dx = self.df.index.freq.nanos / 1.0e9 / 60.0e0
    #     day_dfs = []
    #     groups = self.df.groupby(self.df[[meas_col, model_col]].index.date)
    #     my_list = [day.copy() for (name, day) in groups]
    #     # for name, day in self.df.groupby(self.df[[meas_col, model_col]].index.date):
    #     for day in my_list:
    #         alpha = 1
    #         converged = False
    #         for _ in range(n_iter):
    #             utils.calc_all_window_metrics(day, window, meas_col=meas_col, model_col=model_col, overwrite=overwrite)
    #             X = day[feature_cols].values
    #             y_pred = clf.predict(X)
    #             clear_meas = day[y_pred][meas_col]
    #             clear_model = day[y_pred][model_col]
    #             alpha_last = alpha
    #             def rmse(alpha):
    #                 sqr_err = (clear_meas - (alpha * clear_model))**2
    #                 return np.sqrt(np.mean(sqr_err))
    #             min_scalar = optimize.minimize_scalar(rmse)
    #             alpha = min_scalar.x
    #             day.loc[:, ml_label] = y_pred
    #             if alpha > 1.25 or alpha < .75:
    #                 warnings.warn('Large scaling value.  Day will not be further assessed or scaled.', RuntimeWarning)
    #                 break
    #             day.loc[:, model_col] = day[model_col] * alpha
    #             if np.abs(alpha - alpha_last) < tol:
    #                 converged = True
    #                 break
    #         if not converged:
    #             warnings.warn('Scaling did not converge.', RuntimeWarning)
    #         day_dfs.append(day)
    #     indices = self.df.index
    #     self.df = pd.concat(day_dfs)
    #     self.df.index = indices
    #     return self.df[ml_label]

    def iter_predict_daily(self, feature_cols, meas_col, model_col, clf, window, n_iter=20,
                           tol=1.0e-8, ml_label='sky_status iter', smooth=False, overwrite=True, multiproc=True):
        """Predict clarity based using classifier that iteratively fits the model column
        to the measured column based on clear points.

        This method differs from iter_predict method because it predicts/scales on individual days, not the entire data set.
        This function WILL overwrite columns that already exist in the data frame.

        Method should be refactored.  Calculating features and scaling model GHI is done across entire data set even though we only need one day at a time.

        Arguments
        ---------
        feature_cols: list-like
            Column names to use as features in ML model.
        clf: sklearn estimator
            Object with fit and predict methods.
        meas_col: str
            Column of measured data.
        model_col: str
            Column of model data.
        n_iter, optional: int
            Number of iterations for fitting model to measured column.
        tol, optoinal: float
            Criterion for convergence of modeled and measured clear points.
        ml_label, optional: str
            Label for predicted clear/cloudy points.
        smooth, optional: bool
            Smooth results.  Smoothing is aggressive as a point must be clear
            in every window it appears in.
        overwrite, optional: bool
            Permission to overwrite columns if they exist.
        """
        # day_dfs = []
        final_dfs = []
        groups = self.df.groupby(self.df[[meas_col, model_col]].index.date)
        my_list = [day.copy() for (name, day) in groups]
        if multiproc:
            # pool = mp.Pool(processes=4)
            # day_dfs = [pool.apply_async(utils.day_prediction, args=(day, clf, feature_cols, window, meas_col, model_col),
            #                                               kwds={'overwrite': overwrite, 'n_iter': n_iter, 'tol': tol, 'ml_label': ml_label}) for day in my_list]
            # final_dfs = [d.get() for d in day_dfs]
            # pool.close()
            # pool.join()
            # procs = [mp.Process(target=utils.day_prediction, args=(day, clf, feature_cols, window, meas_col, model_col),
            #                                                  kwargs={'overwrite'=overwrite, 'n_iter'=n_iter, 'tol'=tol, 'ml_label'=ml_label}) for day in my_list]

            # with warnings.catch_warnings():
            #     warnings.simplefilter('ignore')
            with cf.ProcessPoolExecutor() as executor:
                day_dfs = [executor.submit(utils.day_prediction, day, clf, feature_cols,
                                           window, meas_col, model_col, overwrite=overwrite, n_iter=n_iter, tol=tol, ml_label=ml_label) for day in my_list]
                for future in cf.as_completed(day_dfs):
                    final_dfs.append(future.result())
        else:
            final_dfs = [utils.day_prediction(day, clf, feature_cols, window, meas_col, model_col,
                         overwrite=overwrite, n_iter=n_iter, tol=tol, ml_label=ml_label) for day in my_list]
        indices = self.df.index
        self.df = pd.concat(final_dfs)
        self.df.index = indices
        return self.df[ml_label]

    def smooth_ml_label(self, window, label):
        """Smooth labeled points.

        Labeled points are smooth by assuring that each point in a window is labeled as clear.
        This is only appropriate for binary data.
        """
        ser = self.df[label].rolling(window, center=True).apply(lambda x: np.sum(x) == len(x)).fillna(0)
        self.df[label] = ser

    def time_from_solar_noon(self, col, label):
        """Calculate distance from solar noon (absolute) in minutes.

        Solar noon is defined as the index of the peak of the self.df[col] data.

        Arguments
        ---------
        col: str
            Column to be used for solar noon peak finding.
        label:
            Resultant label.
        """
        mins = []
        for name, day in self.df.groupby(self.df.index.date):
            maxidx = day[col].idxmax()
            mins.append(np.asarray((day.index - maxidx).total_seconds()) / 60.0e0)
        mins = np.asarray(mins)
        self.df[label] = mins.flatten()
