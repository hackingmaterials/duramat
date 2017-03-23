"""API wrapper for the PVDAQ database

This API wrapper has the capability to get raw and aggregate
PV data for a specific site given a specific set of dates.

"""

import requests
import pandas as pd


class PVDAQ_API(object):
    """

    class to handle API requests to PVDAQ database

    """

    def __init__(self, api_key):
        """
        initialize base url, api key, and format of responses (forced to be json now)
        """
        self.base_url_ = 'http://developer.nrel.gov/api/pvdaq/v3/'
        self.api_key_ = api_key
        self.form_ = 'json'

    def sites(self, options, result_dict=True):
        """call API to get site metadata info

        Get site metadata which includes:
            comments (str): notes about site
            confidential (bool): is site confidential
            inverter_mfg (str): inverter manufacturer name
            inverter_mfg_hide (bool): is inverter manufacturer confidential
            inverter_model (str): model no. of inverter
            inverter_model_hide (bool): is inverter model confidential
            module_mfg (str): pv module manufacturer
            module_mfg_hide (bool): is module manufacturer confidential
            module_model (str): model no. of module
            module_model_hide (bool): is module model confidential
            site_power (float): power of site (kW?)
            site_power_hide (bool): is site power confidential
            module_tech (int): classification of module technology
            name_private (str): name of site
            name_public (str): name of site
            site_area (float): site of area (m^2)
            site_azimuth (float): azimuthal angle of site (degrees)
            site_elevation (float): elevation of site (m?)
            site_latitude (float): latitude of site (degrees)
            site_longitude (float): longitude of site (degrees)
            site_title (float): tilt of pv site (degrees)
            system_id (int): unique id of site


        Parameters:
            options (dict)
                parameters for api call
                acceptable options are:
                    -- required --
                    + api_key (str) (already included)
                    + system_id (int) (this is listed as optional, but appears to be required...)
                        allowed ids = [2, 3, 4, 10, 17, 18, 33, 34, 35, 38,
                                       39, 50, 51, 1199, 1207, 1208, 1229,
                                       1230, 1231, 1236, 1239, 1276, 1277,
                                       1278, 1283, 1284, 1289, 1292, 1332,
                                       1389, 1403, 1405, 1422, 1423, 1424,
                                       1425, 1429]
                    -- optional --
                    None
            result_dict (bool)
                return either dict of results or full json response

        returns:
            dict (of json response) or dict of output data
        """
        url = self.base_url_ + 'sites.' + self.form_
        options['api_key'] = self.api_key_
        required = ['api_key', 'system_id']
        for requirement in required:
            if requirement not in options.keys():
                raise KeyError('missing required field in options: ' + requirement)
        allowed_system_ids = [2, 3, 4, 10, 17, 18, 33, 34, 35, 38,
                              39, 50, 51, 1199, 1207, 1208, 1229,
                              1230, 1231, 1236, 1239, 1276, 1277,
                              1278, 1283, 1284, 1289, 1292, 1332,
                              1389, 1403, 1405, 1422, 1423, 1424,
                              1425, 1429]
        if int(options['system_id']) not in allowed_system_ids:
            raise KeyError('invalid sytem_id, must be one of the following: ' + ','.join(allowed_system_ids))
        options['system_id'] = str(options['system_id'])
        response = self.base_api_call_(url, options)
        if result_dict:
            return response['outputs'][0]
        else:
            return response

    def raw(self, options, to_dataframe=True):
        """
        call api to get raw data for given site

        arguments:
            options (dict)
            parameters for api call
                acceptable options are:
                    -- required --
                    + api_key (str) (already included)
                    + system_id (int)
                    + start_date (str) (MM/DD/YYYY)
                    + end_date (str) (MM/DD/YYYY)
                    -- optional --
                    UPDATE ME
            to_dataframe (bool)
                convert response to pandas dataframe (of data only)

        returns:
            either dict (of json response) or dataframe
        """
        url = self.base_url_ + 'data.' + self.form_
        options['api_key'] = self.api_key_
        required = ['api_key', 'system_id', 'start_date', 'end_date']
        for requirement in required:
            if requirement not in options.keys():
                raise KeyError('missing required field in options: ' + requirement)
        options['system_id'] = str(options['system_id'])
        response = self.base_api_call_(url, options)
        if to_dataframe:
            return pd.DataFrame(data=response['outputs']['data'][1:], columns=response['outputs']['data'][0])
        else:
            return response

    def aggregate(self, options, to_dataframe=True):
        """
        call api to get raw data for given site

        arguments:
            options (dict)
                parameters for api call
                acceptable options are:
                    -- required --
                    + api_key (str) (already included)
                    + system_id (int)
                    + start_date (str) (MM/DD/YYYY)
                    + end_date (str) (MM/DD/YYYY)
                    + aggregate (str) ("hourly", "daily", "weekly", or "monthly")
                    -- optional --
                    UPDATE ME
            to_dataframe (bool)
                convert response to pandas dataframe (of data only)

        returns:
            either dict (of json response) or dataframe
        """
        url = self.base_url_ + 'site_data.' + self.form_
        options['api_key'] = self.api_key_
        required = ['api_key', 'system_id', 'start_date', 'end_date', 'aggregate']
        for requirement in required:
            if requirement not in options.keys():
                raise KeyError('missing required field in options: ' + requirement)
        options['system_id'] = str(options['system_id'])
        response = self.base_api_call_(url, options)
        if to_dataframe:
            return pd.DataFrame(data=response['outputs'][1:], columns=response['outputs'][0])
        else:
            return response

    @staticmethod
    def base_api_call_(url, options):
        """
        make call to api

        arguments:
            url (str)
                url destination for api call
            options (dict)
                options for api call
            to_dataframe (bool)
                convert response to pandas dataframe (of data only)
        """
        try:
            response = requests.get(url, params=options)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as error:
            print(error)
