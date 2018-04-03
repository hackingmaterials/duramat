
import copy
import warnings
import os
import concurrent.futures as cf
import multiprocessing as mp

import numpy as np
from scipy import optimize
import pandas as pd
import pvlib
import utils
from sklearn import metrics

def main():
    pass


class ClearskyDetection(object):
    """Class for detecting clearsky based on NSRDB data."""

    def __init__(self, df, meas_col, model_col=None, target_col=None, solar_noon_col=None, copy=True):
        """Initialize members.

        Parameters
        ---------
        df: pd.DataFrame
            Time series irradiance data.
        meas_col: str
            Column of measured GHI values.
        model_col: str
            Column of clear sky GHI values.
        target_col: str
            Column of clear sky status.
        copy: bool
            Permission to copy dataframe.
        """
        self.meas_col = meas_col
        self.model_col = model_col
        self.target_col = target_col

        if copy:
            self.df = df.copy()
        else:
            self.df = df

        if target_col is not None:
            self.df[self.target_col] = self.df[self.target_col].astype(bool)

        self.window = self.calc_window()

        self.masks_ = []
        self.features_ = ['avg(GHI)-avg(GHIcs)',
                          'max(GHI)-max(GHIcs)',
                          'GHILL-GHIcsLL',
                          'std(GHI)-std(GHIcs) normed',
                          'max(abs(diff(GHI)-diff(GHIcs)))',
                          'GHI>0']
                          # 't-tnoon']

    @classmethod
    def read_nsrdb_dir(cls, dir_path, timezone, keepers=('GHI', 'Clearsky GHI', 'Cloud Type'), file_ext='csv'):
        """Read directory of NSRDB files.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

        Parameters
        ---------
        dir_path: str
            Path to directory of files.
        timezone: pytz.timezone
            Timezone for the dataframe indicies.
        file_ext: str
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
        df = df.sort_index()
        df = df.reindex(pd.date_range(start=df.index[0], end=df.index[-1], freq='30min')).fillna(0)
        df = df[~df.index.duplicated(keep='first')]
        df.index = df.index.tz_convert('UTC')
        return cls(df[keepers])

    @classmethod
    def read_snl_rtc(cls, file_w_path, timezone, keepers=('GHI')):
        """Read SNL RTC data into file.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

        Parameters
        ----------
        file_w_path: str
            Path to file (absolute).
        timezone: pytz.timezone or str
            Timezone for localization.

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
    def read_srrl_dir(cls, dir_path, timezone='MST', keepers=('GHI')):
        """Read directory of SRRL files into a dataframe.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

        Parameters
        ---------
        dir_path: str
            Path to directory of files.
        timezone: pytz.timezone
            Timezone for the dataframe indicies.
        keepers: list-like
            Columns to keep in dataframe.

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
    def read_ornl_file(cls, filename, timezone='EST', keepers = ('GHI')):
        """Read directory of SRRL files into a dataframe.

        *** NOTE ***
        This is hardcoded for the files I have.  It is not guaranteed to be general at all for
        SRRL/MDIC/etc data sources and files.

        Parameters
        ---------
        filename: str
            Path to directory of files.
        timezone: pytz.timezone
            Timezone for the dataframe indicies.
        keepers: list-like
            Columns to keep in dataframe.

        Returns
        -------
        df: pd.DataFrame
            Contains all fields from files.
        """
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
    def read_pickle(cls, filename, *args, **kwargs):
        """Read dataframe from pickle file.

        Parameters
        ---------
        filename: str
            Name of pickle file.
        """
        df = pd.read_pickle(filename)
        return cls(df, *args, copy=False, **kwargs)

    def calc_window(self):
        dt = np.diff(self.df.index.values)
        dt = float(dt[0] / 1.0e9 / 60.0e0)
        window = int((60 // dt) + 1)
        return window

    def to_pickle(self, filename, overwrite=False):
        """Dump dataframe to pickle file.

        Parameters
        ---------
        filename: str
            Name of pickle file.
        overwrite: bool
            Overwrite file if it exists.
        """
        if os.path.exists(filename) and not overwrite:
            raise FileExistsError('The specified file already exists.  Change name or set overwrite.')
        self.df.to_pickle(filename)

    def robust_rolling_smooth(self, col, smoothing_window, label=None, overwrite=False):
        """Smooth column of data robustly (median + mean).

        Parameters
        ---------
        col: str
            Column in dataframe.
        window: int
            Number of points to include in window.
        label: str
            Name of smoothed column.
        overwrite: bool
            Overwrite new column or not.
        """
        if label is None:
            label = col + ' smooth'
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Provide label name or allow for overwrite.')
        self.df[label] = \
            self.df[col].rolling(smoothing_window, center=True).median().fillna(0).\
                rolling(smoothing_window, center=True).mean().fillna(0)

    def trim_dates(self, x0=None, x1=None):
        """Trim date range of dataframe.

        The x0 is inclusive, x1 is exclusive.  If either are None, assume no trimming in said direction.

        Parameters
        ---------
        x0: datetime or str
            Earlier date (inclusive).
        x1: datetime or str
            Later date (exclusive).
        """
        if x0 is not None:
            self.df = self.df[self.df.index >= x0]
        if x1 is not None:
            self.df = self.df[self.df.index < x1]

    def intersection(self, other_indices):
        """Modify indices such that dataframe intersects with different set of indices.

        Parameters
        ---------
        other_indices: pd.DateTimeIndex
            Indices to intersect with.
        """
        indices = self.df.index.intersection(other_indices)
        self.df = self.df[self.df.index.isin(indices)]

    def by_time_of_day_transform(self, data='GHI'):
        """Transform time series (linear) to matrix representation.

        The dataframe will be organized as Date (columnss) X Timestample (rows).

        Parameters
        ---------
        data: str
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

        Parameters
        ---------
        data: str
            Data column to generate clearsky, assumed to be 'GHI'.
        num_days: int
            Size of window (in days) for constructing clearsky curve.
        model_fxn: callable
            Function for determining clearsky curve.
        percentile: float
            Perctile value used in statistical clearsky, ignored if model_fxn is not np.percentile or np.nanpercentile.
        smooth_window: int
            Size (number of data points) for smoothing the statistical clear sky.
        smooth_fxn: callable
            Function for smoothing the clear sky curve.
        label: str
            Name of statistical clearsky column for dataframe.
        overwrite: bool
            Permission to overwrite existing columns.
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

        Parameters
        ---------
        dates: list-like
            Unique dates in data.
        day_to_filter: datetime.date
            Day which will be filtered.
        sample_days: pd.DataFrame
            DataFrame of measured values (rows are time of day and columns are dates).
        model_fxn: callable
            Function that will be used to construct the statistical clearsky curve.
        percentile: float
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

        Parameters
        ----------
        dates: list-like
            Unique dates in data.
        num_days: int
            Size of window in days.

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

    def generate_pvlib_clearsky(self, latitude, longitude, altitude=None, tz=None,
                                label='Clearsky GHI pvlib', overwrite=False):
        """Generate pvlib clearsky curve.

        Parameters
        ----------
        latitude: float
            Latitutde.
        longitude: float
            Longitude.
        altitude: float
            Elevation of system (this is not required by Location object).
        label: str
            Label of clearsky data points.
        overwrite: bool
            Allow column to be overwritten if exists.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        # loc = pvlib.location.Location(latitude, longitude, altitude=altitude)
        loc = pvlib.location.Location(latitude, longitude, tz=tz, altitude=altitude)
        clear_skies = pd.DataFrame(loc.get_clearsky(self.df.index), index=self.df.index)
        # clear_skies.index = self.df.index
        # clear_skies.index = clear_skies.index.tz_localize(tz)
        self.df[label] = clear_skies['ghi']
        self.model_col = label

    def pvlib_clearsky_detect(self, scale=False, measured='GHI', modeled='Clearsky GHI pvlib', label='sky_status pvlib',
                              window=10, pvlib_kwargs={}, overwrite=False):
        """Detect clearsky using PVLib methods.

        Parameters
        ----------
        scale: bool
            Scale modeled column by factor determined by PVLib.
        measured: str
            Column to detect.
        modeled: str
            Column of modeled clearsky (reference for measured).
        label: str
            Classification label name.
        window: int
            Size of window for PVLib detection.
        overwrite: bool
            Allow column to be overwritten if label exists.
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        alpha = 1.0
        if scale:
            status, _, alpha = \
                pvlib.clearsky.detect_clearsky(self.df[measured], self.df[modeled],
                                               self.df.index, window, **pvlib_kwargs, return_components=True)
        else:
            status = pvlib.clearsky.detect_clearsky(self.df[measured], self.df[modeled], self.df.index,
                                                    window, **pvlib_kwargs)
        self.df[label] = status
        if scale:
            self.df[modeled + ' scaled'] = alpha * self.df[modeled]

    def set_nsrdb_sky_status(self, label='sky_status', overwrite=False):
        """Set sky status target for machine learning applications for nsrdb data.

        Parameters
        ----------
        label: str
            Column name for new sky status.
        overwrite: bool
            Allow column to be overwritten in column exists
        """
        if label in self.df.keys() and not overwrite:
            raise RuntimeError('Label already exists.  Set overwrite to True or pick new label name.')
        self.df[label] = (self.df['Cloud Type'] == 0) & (self.df['GHI'] > 0)

    def calc_all_metrics(self):
        """Wrapper function for utils.calc_all_window_metrics for investigating features.
        """
        utils.calc_all_window_metrics(self.df, self.window, self.meas_col, self.model_col, overwrite=True)
        self.time_from_solar_noon()
        # for feat in [i for i in self.features_ if i not in ('t-tnoon', 'std(GHI)-std(GHIcs) normed')]:
        #     self.df[feat] = (self.df[feat] / self.df['GHI mean']).replace([np.nan, np.inf, -np.inf], 0)

    def calc_window_stats(self, col, overwrite=False):
        """Caclulates mean, std, min, max across all windows for specified column.

        Parameters
        ----------
        col: str
            Name of column.
        overwrite: bool
            Allow labels to be overwritten if they already exist.

        Returns
        -------
        None - modifies in place.
        """
        utils.calc_window_stat(self.df, self.window, col, col, overwrite=overwrite)

    def scale_model(self):
        """Scale model values to measured values based on clear sky periods.

        """
        # for day, group in self.df.groupby([self.df.index.year, self.df.index.week]):
        # for day, group in self.df.groupby([self.df.index.year, self.df.index.week]):
        #     mask = group[self.target_col].astype(bool).values & np.all(group[self.masks_].values, axis=1)
        #     if np.sum(mask) == 0:
        #         continue
        #     clear_meas = group[mask][self.meas_col]
        #     clear_model = group[mask][self.model_col]

        #     def rmse(alpha):
        #         sqr_err = (clear_meas - (alpha * clear_model))**2
        #         return np.sqrt(np.mean(sqr_err))

        #     alp = optimize.minimize_scalar(rmse).x
        #     self.df.loc[self.df.index.isin(group.index), self.model_col] = alp * group[self.model_col]

        mask = self.df[self.target_col].astype(bool).values & np.all(self.df[self.masks_].values, axis=1)
        # if np.sum(mask) == 0:
        #     continue
        clear_meas = self.df[mask][self.meas_col]
        clear_model = self.df[mask][self.model_col]

        def rmse(alpha):
            sqr_err = (clear_meas - (alpha * clear_model))**2
            return np.sqrt(np.mean(sqr_err))

        alp = optimize.minimize_scalar(rmse).x
        self.df.loc[:, self.model_col] = alp * self.df[self.model_col]

    def fit_model(self, clf, scale=True, clean=False, sample_weight=None, *args, **kwargs):
        """Fit an sklearn estimator object.

        Parameters
        ----------
        clf: estimator object
            Must have fit method.
        feature_cols: str
            Column names of features to use in fit.
        scale: bool
            Scale model before fit.

        Returns
        -------
        clf: estimator object
            Fitted estimator.

        """
        if scale:
            self.scale_model()
        utils.calc_all_window_metrics(self.df, self.window, self.meas_col, self.model_col, overwrite=True)
        # if clean:
        #     self.filter_labels(*args, **kwargs)  # do not rescale - only remove 'bad' cloudy periods
        #     # self.scale_model()
        #     # utils.calc_all_window_metrics(self.df, self.window, self.meas_col, self.model_col, overwrite=True)
        # else:
        #     self.df['mask'] = True
        self.filter_labels(**kwargs, overwrite=True)
        # clf.fit(self.df[self.df['mask']][self.features_], self.df[self.df['mask']][self.target_col])
        train_df = self.get_masked_df()
        # if sample_weight is not None:
        #     clf.fit(train_df[self.features_].values, train_df[self.target_col].values, train_df[sample_weight].values)
        # else:
        clf.fit(train_df[self.features_].values, train_df[self.target_col].astype(int).values)
        return clf

    def filter_labels(self, ratio_mean_val=None, diff_mean_val=None, mask_label='quality_mask', overwrite=False):
        """Generate mask to remove incorrectly labeled points from training set.

        Resulting mask should remove the cloudy points that 'look' clear.  It should include
        every period originally labeled as clear.

        By default, no filtering is done.  Must actively choose to do so.
        """
        if mask_label in self.masks_ and not overwrite:
            raise RuntimeError('Label already exists as mask.  Change name or allow overwrite.')
        if not all(i in self.df.keys() for i in self.features_):
            self.scale_model()
            utils.calc_all_window_metrics(self.df, self.window, self.meas_col, self.model_col, overwrite=True)
        self.df[mask_label] = True
        if ratio_mean_val is not None and diff_mean_val is not None:
            # self.df.loc[(~self.df['sky_status']) &
            #             ((self.df['GHI/GHIcs mean'] >= ratio_mean_val) |
            #             (np.abs(self.df['GHI-GHIcs mean']) <= diff_mean_val)), mask_label] = False
            self.df.loc[(~self.df['sky_status']) &
                        ((np.abs(1 - self.df['GHI/GHIcs mean']) <= ratio_mean_val) |
                         (np.abs(self.df['GHI-GHIcs mean']) <= diff_mean_val)), mask_label] = False
        if mask_label not in self.masks_:
            self.masks_.append(mask_label)

    def clear_masks(self, which=None):
        """Clear filtering masks.

        Parameters
        ----------
        which: str or list
            Name(s) of columns to be removed as a mask.  If None, all masks removed.
        """
        if type(which) == str:
           which = [which]
        if which is None:
            self.masks_ = []
        else:
            self.masks_ = [i for i in self.masks_ if i not in which]

    def get_masked_df(self):
        """Returns dataframe where all masks are true.

        Returns
        -------
        df: pd.DataFrame
            Dataframe excluding masked rows.
        """
        df = self.df
        # df[self.masks_] = df[self.masks_].fillna(False)
        if self.masks_:
            for mask_col in self.masks_:
                df = df[df[mask_col]]
            # df = np.all(df[self.masks_].values, axis=1)
        return df

    def add_mask(self, mask_name, mask_array, fillval=False, overwrite=False):
        """Add bool mask to dataframe.

        Parameters
        ----------
        mask_name: str
            Name to use for mask column.
        mask_array: np.array
            Bool array.
        """
        if mask_name in self.df.keys() and not overwrite:
            raise RuntimeError('Mask name already exists.  Allow overwriting or rename.')
        self.df[mask_name] = mask_array
        self.df[mask_name] = self.df[mask_name].fillna(fillval)
        self.masks_.append(mask_name)

    def iter_predict_daily(self, clf, n_iter=20, tol=1.0e-8, ml_label='sky_status iter',
                           overwrite=True, multiproc=True, by_day=True, proba_cutoff=None):
        """Predict clarity based using classifier that iteratively fits the model column
        to the measured column based on clear points.

        This method differs from iter_predict method because it predicts/scales on individual days,
        not the entire data set.  This function WILL overwrite columns that already exist in the data frame.

        Parameters
        ----------
        clf: sklearn estimator
            Object with fit and predict methods.
        n_iter: int
            Number of iterations for fitting model to measured column.
        tol: float
            Criterion for convergence of modeled and measured clear points.
        ml_label: str
            Label for predicted clear/cloudy points.
        overwrite: bool
            Permission to overwrite columns if they exist.
        multiproc: bool
            Use multiprocessing (only works if by_day is also True).
        by_day: bool
            Group data set by day and make predictions on single days (as opposed to entire data set at once).
        """
        final_dfs = []
        if by_day:
            groups = self.df.groupby([self.df.index.year, self.df.index.week])
            my_list = [day.copy() for (name, day) in groups]
        else:
            my_list = [self.df]
        if multiproc:
            def append_result(res):
                final_dfs.append(res)

            pool = mp.Pool()
            for day in my_list:
                pool.apply_async(
                    utils.day_prediction, (day, clf, self.features_, self.window, self.meas_col, self.model_col),
                    {'overwrite': overwrite, 'n_iter': n_iter, 'tol': tol,
                     'ml_label': ml_label, 'proba_cutoff': proba_cutoff}, callback=append_result
                )
            pool.close()
            pool.join()
        else:
            final_dfs = [utils.day_prediction(day, clf, self.features_, self.window, self.meas_col, self.model_col,
                         overwrite=overwrite, n_iter=n_iter, tol=tol, ml_label=ml_label) for day in my_list]
        if multiproc:
            self.df = pd.concat(final_dfs).sort_index()
        else:
            self.df = pd.concat(final_dfs)
        self.df.index = pd.to_datetime(self.df.index)
        self.df.loc[self.df['GHI'] <= 0, ml_label] = False
        return self.df[ml_label]

    def predict(self, clf, ml_label='sky_status iter', proba_cutoff=.5):
        self.calc_all_metrics()
        X = self.df[self.features_].values
        try:
            probas = clf.predict_proba(X)
            pred = probas[:, 1] >= proba_cutoff
            self.df['probas'] = probas[:, 1]
        except AttributeError:
            pred = clf.predict(X)
        self.df[ml_label] = pred
        self.df.loc[self.df['GHI'] <= 0, ml_label] = False
        return self.df[ml_label]

    def downsample(self, min_freq):
        """Samples data frame at every min_freq step.  For example, if self.df is minutely data and every 30th minute
        is desired, pass min_freq = 30.  This is different than pandas resample in that we do not apply a
        smoothing/aggregating/statistical function.

        This will modify self.df irreversibly.

        Parameters
        ----------
        min_freq: int, float
            Which minute of the hour to downsample to.

        """
        if min_freq < 1 or min_freq > 60:
            raise NotImplementedError('Can only downsample to between 1 and 60 minutes right now.')
        if float(int(min_freq)) != min_freq or type(min_freq) != int:
            raise RuntimeError('Must pass an integer between 1 and 60.')
        mask = self.df.index.minute % min_freq == 0
        self.df = self.df[mask]
        # self.window = self.calc_window()

    def time_from_solar_noon(self):
        """Calculate minutes until solar noon for each point.  Only finds solar noon in a given day, which
        should be the closest solar noon.
        """
        dt = np.diff(self.df.index.values)
        dt = float(dt[0] / 1.0e9 / 60.0e0)
        day_window = int(24 * 60 / dt)
        tstamps = self.df.index.values
        # indices = np.asarray([np.arange(max(0, i - (day_window // 2)),
        #                                 min(len(tstamps), i + (day_window // 2)))
        #                       for i in range(len(tstamps))])
        # time_steps = []
        # for i, ix in enumerate(indices):
        #     argmax = np.argmax(self.df[self.model_col].values[ix])
        #     time_steps.append(np.abs(i - (argmax + ix[0])))
        # times = np.asarray(time_steps) * dt
        # label = 'abs(t-tnoon)'
        # self.df[label] = times.flatten()

        def indices_gen():
            for i in range(len(tstamps)):
                yield np.arange(max(0, i - (day_window // 2)), min(len(tstamps), i + (day_window // 2)))

        ts = np.zeros(len(tstamps), dtype=int)
        vals = self.df[self.model_col].values.astype(int)
        label = 't-tnoon'
        # print(indices.nbytes / 1e6)
        # print(ts.nbytes / 1e6)
        # print(vals.nbytes / 1e6)
        for i, ix in enumerate(indices_gen()):
            argmax = np.argmax(vals[ix])
            # ts[ix] = np.abs(i - (argmax + ix[0])) * dt
            ts[i] = np.abs(i - (argmax + ix[0])) * dt
            # ts[i] = np.abs(i - (argmax + ix[0])) * dt
        # times = np.asarray(time_steps) * dt
        self.df[label] = ts

    def get_features_and_targets(self, nsrdb_obj, tsplit=None, ignore_nsrdb_mismatch=False):
        # filter and preprocess data
        #   fill values < 0 to 0
        #   add in target column
        #   remove missing days
        #   remove periods with missing cloud data
        #   remove periods where nsrdb and ground disagree
        nsrdb_obj.intersection(self.df.index)
        self.add_target_col(nsrdb_obj)
        self.add_data_col(nsrdb_obj, 'GHI')
        self.scale_model()

        self.fill_low_to_zero()
        self.mask_missing_days()
        self.mask_night()
        self.mask_missing_clouds(nsrdb_obj)
        self.mask_nsrdb_incorrect_clouds(nsrdb_obj)
        self.mask_nsrdb_incorrect_clear(nsrdb_obj)
        self.mask_maybe_clear(nsrdb_obj)
        if not ignore_nsrdb_mismatch:
            self.mask_nsrdb_mismatch(nsrdb_obj)

        # mask for splitting by time (useful for CV, train/test splitting)
        if tsplit is not None:
            self.df['before'] = self.df.index < tsplit
        else:
            self.df['before'] = True

        # calculate window based ML metrics
        self.calc_all_metrics()

        # return feature matrix and target vector as dataframes
        return self.df[self.features_], self.df[self.target_col], self.df[self.masks_].all(axis=1), self.df['before']

    def add_target_col(self, nsrdb_obj, name='sky_status'):
        """Add in a target column (for classification).

        Parameters
        ----------
        nsrdb_obj: ClearskyDetection object
            Must have 'sky_status' column (or 'Cloud Type' and 'GHI' from which it can be derived).
        name: str
            Name of target column

        Returns
        -------
        None
        """
        self.target_col = name
        if name not in nsrdb_obj.df.keys():
            nsrdb_obj.set_nsrdb_sky_status()
        self.df[self.target_col] = nsrdb_obj.df[name].astype(int)
        self.df[self.target_col] = self.df[self.target_col].fillna(0).astype(int)

    def add_data_col(self, nsrdb_obj, col):
        self.df[col + ' nsrdb'] = nsrdb_obj.df[col]
        # self.df[col + ' nsrdb'] = self.df[col + ' nsrdb'].interpolate()

    def fill_low_to_zero(self, low_cutoff=0):
        """Make GHI values below zero equal to zero.

        Parameters
        ----------
        low_cutoff

        Returns
        -------

        """
        self.df.loc[self.df[self.meas_col] < low_cutoff, self.meas_col] = 0
        self.df[self.meas_col] = self.df[self.meas_col].fillna(0)

    def mask_missing_days(self, daily_cutoff=200):
        """Mask entire days where total irradiance is too low.
        """
        resample = self.df[self.meas_col].resample('D').sum()
        bad_dates = resample[resample < daily_cutoff].index.date
        mask = ~np.isin(self.df.index, bad_dates)
        self.add_mask('empty_days', mask, overwrite=True)

    def mask_missing_clouds(self, nsrdb_obj):
        """Mask periods missing cloud data (Cloud Type < 0).
        """
        mask = nsrdb_obj.df['Cloud Type'].astype(int) >= 0
        self.add_mask('has_clouds', mask, overwrite=True)

    def mask_irrad(self, cutoff=200):
        """Mask low irradiance values.
        """
        mask = self.df['GHI'] >= 200
        self.add_mask('low_irrad', mask)

    def mask_nsrdb_incorrect_clouds(self, nsrdb_obj, diff_mean_val=50, label='nsrdb_cloud_quality'):
        """Generate mask to remove incorrectly labeled points from training set.

        Resulting mask should remove the cloudy points that 'look' clear.  It should include
        every period originally labeled as clear.

        By default, no filtering is done.  Must actively choose to do so.
        """
        if 'sky_status' not in nsrdb_obj.df.keys():
            nsrdb_obj.set_nsrdb_sky_status()
        nsrdb_obj.scale_model()
        nsrdb_obj.df[label] = True
        nsrdb_obj.df['GHI mean'] = nsrdb_obj.df['GHI'].rolling(3, center=True).mean()
        nsrdb_obj.df['GHIcs mean'] = nsrdb_obj.df['Clearsky GHI pvlib'].rolling(3, center=True).mean()
        if diff_mean_val is not None:
            nsrdb_obj.df.loc[(~nsrdb_obj.df['sky_status']) &
                             (np.abs(nsrdb_obj.df['GHI mean'] - nsrdb_obj.df['GHIcs mean']) <= diff_mean_val) &
                             # (np.abs(nsrdb_obj.df['GHI'] - nsrdb_obj.df['Clearsky GHI pvlib']) <= diff_mean_val) &
                             (nsrdb_obj.df['GHI'] > 0), label] = False
        self.add_mask(label, nsrdb_obj.df[label], overwrite=True)

    def mask_nsrdb_incorrect_clear(self, nsrdb_obj, diff_mean_val=50, label='nsrdb_clear_quality'):
        """Mask periods where NSRDB labeled points as clear even though GHI and GHIcs are 'too different'.
        """
        nsrdb_obj.scale_model()
        nsrdb_obj.df[label] = True
        nsrdb_obj.df['GHI mean'] = nsrdb_obj.df['GHI'].rolling(3, center=True).mean()
        nsrdb_obj.df['GHIcs mean'] = nsrdb_obj.df['Clearsky GHI pvlib'].rolling(3, center=True).mean()
        nsrdb_obj.df.loc[(nsrdb_obj.df['sky_status']) &
                         (np.abs(nsrdb_obj.df['GHI mean'] - nsrdb_obj.df['GHIcs mean']) > diff_mean_val), label] = False
                         # (np.abs(nsrdb_obj.df['GHI'] - nsrdb_obj.df['Clearsky GHI pvlib']) > diff_mean_val), label] = False
        self.add_mask(label, nsrdb_obj.df[label], overwrite=True)

    def mask_nsrdb_mismatch(self, nsrdb_obj, diff_threshold=50, label='nsrdb_mismatch'):
        """Add filter to mask periods when NSRDB and ground measurements are 'too different'.
        """
        window_size_dict = {30: 3, 15: 5, 10: 7, 5: 13, 1: 61}
        # indices = self.df.index.intersection(nsrdb_obj.df.index)
        nsrdb_ghi = nsrdb_obj.df['GHI']
        ground_ghi = self.df['GHI']

        freq = int(np.unique(np.diff(self.df.index))[0] / 1.0e9 / 60.0e0)

        nsrdb_ghi_mean= nsrdb_ghi.rolling(3, center=True).mean().fillna(0)
        ground_ghi_mean= ground_ghi.rolling(window_size_dict[freq], center=True).mean().fillna(0)

        mask = np.abs(nsrdb_ghi_mean - ground_ghi_mean) <= diff_threshold
        self.add_mask(label, mask, overwrite=True)

    def mask_maybe_clear(self, nsrdb_obj):
        """Mask probably clear label from NSRDB"""
        self.add_mask('probably_clear', ~(nsrdb_obj.df['Cloud Type'].astype(int) == 1))

    def mask_night(self):
        self.add_mask('day_time', self.df['GHI'] > 0)

    def cross_val_score(self, clf, cv=5, filter_fit=False, filter_score=False, scoring='accuracy', filter_kwargs=None,
                        fit_args=None, fit_kwargs=None, predict_args=None, predict_kwargs=None,
                        scoring_args=None, scoring_kwargs=None):
        """Perform cross validation on self.df data.

        Parameters
        ----------
        clf: sklearn estimator
            Classifier to perform CV with.
        cv: int
            Number of CV folds.
        scoring: str
            Which scoring method to use.  Options are ('f1', 'accuracy', 'precision', 'recall').
        filter_fit: bool
            Filter training data based on filterkwargs before fitting model.
        filter_score: bool
            Filter test data based on filterkwargs after prediction but before scoring.
        fit_args: list
            Positional arguments for fitting.
        fit_kwargs: dict
            Keyword arguments for fitting.
        predict_args: list
            Positional arguments for prediction.
        predict_kwargs: dict
            Keyword arguments for prediction.
        scoring_args: list
            Positional arguemnts for scoring.
        scoring_kwargs: dict
            Keyword arguemnts for scoring.

        Returns
        -------
        scores: list
            Scores for each fold.

        """
        if fit_args is None:
            fit_args = []
        if fit_kwargs is None:
            fit_kwargs = {}
        if predict_args is None:
            predict_args = []
        if predict_kwargs is None:
            predict_kwargs = {}
        if scoring_args is None:
            scoring_args = []
        if scoring_kwargs is None:
            scoring_kwargs = {}
        if filter_kwargs is None:
            filter_kwargs = {}
        unique_dates = pd.unique(self.df.index.date)
        chunk_size = int(np.ceil(len(unique_dates) / cv))
        scorers_dict = {'f1': metrics.f1_score,
                        'accuracy': metrics.accuracy_score,
                        'precision': metrics.precision_score,
                        'recall': metrics.recall_score}
        scorer = scorers_dict[scoring]
        scores = []
        for i in range(0, len(unique_dates), chunk_size):
            # test_dates = unique_dates[i: i + chunk_size]
            train_dates = np.concatenate((unique_dates[:i], unique_dates[i + chunk_size:]))
            train_mask = np.isin(self.df.index.date, train_dates)
            train_obj = self.__class__(self.df[train_mask], self.meas_col, self.model_col, self.target_col)
            if filter_fit:
                trained_clf = train_obj.fit_model(clf, *fit_args, **fit_kwargs, **filter_kwargs)
            else:
                trained_clf = train_obj.fit_model(clf, *fit_args, **fit_kwargs)
            test_obj = self.__class__(self.df[~train_mask], self.meas_col, self.model_col, self.target_col)
            pred = test_obj.predict(trained_clf, *predict_args, **predict_kwargs)
            if filter_score:
                test_obj.filter_labels(**filter_kwargs)
            else:
                test_obj.df['mask'] = True
            y = test_obj.df[self.target_col][test_obj.df['mask']]
            yhat = pred[test_obj.df['mask']]
            scores.append(scorer(y, yhat, *scoring_args, **scoring_kwargs))
        return scores

    def get_mask_tsplit(self, nsrdb_obj, tsplit=None, ignore_nsrdb_mismatch=False):
        """Function to perform general data cleaning and masking to remove errors in ground and satellite data.

        filter and preprocess data
            fill values < 0 to 0
            add in target column
            remove missing days
            remove periods with missing cloud data
            remove periods where nsrdb and ground disagree 

        Parameters
        ----------
        nsrdb_obj: ClearskyDetection object
        tsplit: string, datetime object
            Date and time where to split data (for training and testing).
        ingore_nsrdb_mismatch: bool
            Ignore filtering mismatched measurements between NSRDB and ground data.

        Returns
        -------
        mask: np.array of bool
            True values are data points that are 'clean enough' for training/scoring.
        tsplit: np.array of bool
            True values are before tsplit date, False values are after.
        """

        nsrdb_obj.intersection(self.df.index)
        self.add_target_col(nsrdb_obj)
        self.add_data_col(nsrdb_obj, 'GHI')
        self.scale_model()

        self.fill_low_to_zero()
        self.mask_missing_days()
        self.mask_night()
        self.mask_missing_clouds(nsrdb_obj)
        self.mask_nsrdb_incorrect_clouds(nsrdb_obj)
        self.mask_nsrdb_incorrect_clear(nsrdb_obj)
        self.mask_maybe_clear(nsrdb_obj)
        if not ignore_nsrdb_mismatch:
            self.mask_nsrdb_mismatch(nsrdb_obj)

        # mask for splitting by time (useful for CV, train/test splitting)
        if tsplit is not None:
            self.df['before'] = self.df.index < tsplit
        else:
            self.df['before'] = True

        # return feature matrix and target vector as dataframes
        return self.df[self.masks_].all(axis=1), self.df['before']



if __name__ == '__main__':
    main()
