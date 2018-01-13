

import os
import warnings
import pandas as pd
import numpy as np
import pvlib

import cs_detection


def main():
    pass


class NSRDBPreprocessor(object):
    """
    Class for preprocessing large batches of NSRDB data.  NSRDB delivers data in large zip files.  After extracting
    them into directories, this object can be used to set them up for the cs_detection machine learning work.


    """
    def __init__(self, path_to_read_dir, path_to_write_dir=None):
        """

        Parameters
        ----------
        path_to_read_dir: str
            Path to directory where extracted NSRDB files.
        path_to_write_dir: str
            Path to where processed files should be written.  Defaults to path_to_read_dir if not given.
        """
        self.path_to_read_dir = path_to_read_dir
        if path_to_write_dir is None:
            self.path_to_write_dir = self.path_to_read_dir
        else:
            self.path_to_write_dir = path_to_write_dir
        self._parse_dir()

    def _parse_dir(self):
        """Make dictionary mapping unique locations to lists of files and populates self.files_df attribute.

        This method assumes that the filenames, as supplied by NSRDB, have not been altered.  Doing so will break this
        function/object.

        The filename format is:
            AAAA_BBBB_CCCC_DDDD.csv
            where AAAA: some identifier
                  BBBB: latitude of site
                  CCCC: longitude of site
                  DDDD: year of data

        Returns
        -------
        None
        """
        contents = [i for i in os.listdir(self.path_to_read_dir) if i.endswith('.csv')]
        files_dict = {}
        for f in contents:
            try:
                metadata = f.split('_')
                metadata[-1] = metadata[-1].replace('.csv', '')
                files_metatdata = {'id': int(metadata[0]),
                                   'lat': float(metadata[1]),
                                   'lon': float(metadata[2]),
                                   'year': float(metadata[3])}
                files_dict[f] = files_metatdata
            except IndexError:
                message = 'Format of filename {} not recognized - ignoring.'.format(f)
                warnings.warn(message, RuntimeWarning)
        self.files_df = pd.DataFrame(files_dict).T

    def combine_files_setup(self, outformat='pkl.gz'):
        """Combine multiple files (for a given site) into a single file data set.  Will also use PVLib get_clearsky
        method and fill in Clearsky GHI pvlib column.

        Returns
        -------
        None
        """
        time_cols = ['Year', 'Month', 'Day', 'Hour', 'Minute']
        for id, file_set in self.files_df.groupby(self.files_df['id']):
            files = [os.path.join(self.path_to_read_dir, f) for f in file_set.index]
            header = pd.read_csv(files[0], nrows=2)  # read header to get time zone, latitude, longitude, elevation
            tz = 'Etc/GMT' + header['Time Zone'][0].replace('-', '+')  # negative sign confuses 'Etc/GMTXX' timezone?
            df = pd.concat([pd.read_csv(f, skiprows=2) for f in files])
            df.index = pd.to_datetime(df[time_cols])
            df.index = df.index.tz_localize(tz)
            df = df.drop(time_cols, axis=1)
            latitude = float(header['Latitude'][0])
            longitude = float(header['Longitude'][0])
            elevation = float(header['Elevation'][0])
            # add Is clear NSRDB column and Clearsky GHI pvlib column
            # Scale Clearsky GHI pvlib to match periods of clarity between
            detection = cs_detection.ClearskyDetection(df, copy=False, set_ghi_status=True)
            detection.set_nsrdb_sky_status(label='Is clear NSRDB')
            detection.generate_pvlib_clearsky(latitude, longitude, elevation, tz=tz)
            detection.scale_model('GHI', 'Clearsky GHI pvlib', 'Is clear NSRDB')
            df = detection.df
            if outformat == 'pkl':
                pd.to_pickle(df, os.path.join(self.path_to_write_dir, str(int(id))) + '.pkl')
            elif outformat == 'pkl.gz':
                pd.to_pickle(df, os.path.join(self.path_to_write_dir, str(int(id))) + '.pkl.gz')
            elif outformat == 'csv':
                df.to_csv(os.path.join(self.path_to_write_dir, str(int(id))) + '.csv')
        print('Files successfully written to {}'.format(self.path_to_write_dir))


if __name__ == '__main__':
    main()
