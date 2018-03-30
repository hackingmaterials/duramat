"""API wrapper for the PVDAQ database

This API wrapper has the capability to get raw and aggregate
PV data for a specific site given a specific set of dates.

"""

import datetime

import requests
import numpy as np
import pandas as pd

def main():
    pass


class PVDAQ_API(object):
    """

    Class to handle API requests to PVDAQ database

    """

    def __init__(self, api_key):
        """initialize object

        Base url and API key are set.  Allowed system_ids also defined here.
        API calls require a valid system ID be set and so far, there seems to be
        no way to fetch them.

        Parameters
        ----------
        api_key: str
            PVDAQ API key
        """
        self.base_url_ = 'http://developer.nrel.gov/api/pvdaq/v3/'
        self.api_key_ = api_key
        # self.allowed_system_ids = [2, 3, 4, 10, 17, 18, 33, 34, 35, 38,
        #                            39, 50, 51, 1199, 1207, 1208, 1229,
        #                            1230, 1231, 1236, 1239, 1276, 1277,
        #                            1278, 1283, 1284, 1289, 1292, 1332,
        #                            1389, 1403, 1405, 1422, 1423, 1424,
        #                            1425, 1429]
        self.allowed_system_ids = (2, 3, 4, 10, 18, 33, 34, 35, 39, 50,
                                   51, 53, 54, 55, 57, 58, 59, 1199, 1200,
                                   1201, 1202, 1203, 1204, 1207, 1208, 1214,
                                   1216, 1218, 1219, 1220, 1221, 1222, 1223,
                                   1224, 1225, 1226, 1229, 1230, 1231, 1232,
                                   1233, 1234, 1235, 1236, 1237, 1239, 1244,
                                   1245, 1246, 1247, 1248, 1249, 1250, 1251,
                                   1252, 1253, 1254, 1255, 1256, 1257, 1258,
                                   1259, 1260, 1261, 1262, 1263, 1264, 1265,
                                   1266, 1267, 1268, 1269, 1270, 1271, 1272,
                                   1273, 1274, 1275, 1276, 1277, 1278, 1283,
                                   1284, 1289, 1292, 1332, 1389, 1403, 1405,
                                   1418, 1419, 1420, 1422, 1423, 1424, 1425,
                                   1429, 1430, 1431, 1432, 1433)


    def sites_metadata(self, r_format='json', api_key=None, system_id=None, user_id=None, raw_response=False):
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

        For allowed system_id values (gathered by hand) see self.allowed_system_ids


        Parameters
        ----------
        r_format: str
            format of response
        api_key: str
            PVDAQ API key
        system_id: int
            id of system of interest
            ** Note that PVDAQ API documentation says this is optional - this seems to be untrue.
               Omitting system_id gives an error.  Investigate further.
        user_id: str, optional
            only for admin accounts.  will return all sites belonging to this id.
        raw_response: bool, optional
            True:
                returns response of requests.get
            False:
                returns site metadata from requests.get response

        Returns
        -------
        result: pandas Series or dict
            contains site metadata (detailed above) or raw json response
        """
        if r_format != 'json':
            raise NotImplementedError('Only json response format is supported at this time.')

        req_params = {}

        req_params['api_key'] = self.api_key_check_(api_key)

        req_params['system_id'] = self.system_id_check_(system_id)

        if user_id is not None:
            req_params['user_id'] = str(user_id)

        url = self.base_url_ + 'sites.' + r_format
        response = self.base_api_call_(url, req_params)

        if raw_response:
            result = response
        else:
            result = pd.Series(response['outputs'][0])

        return result

    def raw_site_data(self, r_format='json', api_key=None, system_id=None, start_date=None, end_date=None,
                      user_id=None, raw_response=False, step_size=14):
        """Get raw data for given site and date range.

        Data includes the following (by default):
            SiteID (int)
            Date-Time (datetime)
            poa_irradiance (float)
            dc_power (float)
            dc_pos_voltage (float)
            cd_pos_current (float)
            module_temp_1 (float)
            das_temp (float)
            das_batter_voltage (float)

        This function performs repeated requests.  The current implementation
        is rather slow.  An obvious improvement is to make the serial requests
        asynchronous/parallel.

        Parameters
        ----------
        r_format: str
            format of response
        api_key: str
            PVDAQ API key
        system_id: int
            id of system of interest
        start_date: str
            begin date for data in MM/DD/YYYY format
            if start_date = 'origin', begin date is 01/01/0001
        end_date: str
            end date for data in MM/DD/YYYY format
            if end_date = 'today', current date will be used
        user_id: str, optional
            only for admin accounts.
            will return all sites belonging to this id.
        raw_response: bool, optional
            True:
                returns all response of requests.get
            False:
                returns desired aggregate data from requests.get response
        step_size: int
            number of days to grab in each call (experiment)

        Returns
        -------
        result: list or dict
            list: responses from response.get calls
            dict: raw data consolidated into single dictionary
        """
        if r_format != 'json':
            raise NotImplementedError('Only json response format is supported at this time.')

        req_params = {}

        req_params['api_key'] = self.api_key_check_(api_key)

        req_params['system_id'] = self.system_id_check_(system_id)

        start_date, end_date = self.date_range_check_(start_date, end_date, system_id)
        if any(i is None for i in (start_date, end_date)):
            return None

        if user_id is not None:
            req_params['user_id'] = str(user_id)

        url = self.base_url_ + 'data.' + r_format

        # cannot get all site data in single block
        # will grab monthly blocks
        date_init = datetime.datetime.strptime(start_date, '%m/%d/%Y').date()
        date_final = datetime.datetime.strptime(end_date, '%m/%d/%Y').date()
        step_size = datetime.timedelta(days=step_size)
        responses = []
        while date_init <= date_final:
            date_add_step = date_init + step_size
            date_add_step = min(date_final, date_add_step)
            date_init_fmat = date_init.strftime('%m/%d/%Y')
            date_add_step_fmat = date_add_step.strftime('%m/%d/%Y')
            req_params['start_date'] = date_init_fmat
            req_params['end_date'] = date_add_step_fmat
            print(date_init_fmat, date_add_step_fmat)
            response = self.base_api_call_(url, req_params)
            if raw_response:
                response.append(response)
            else:
                responses.append(self.parse_response_outputs_to_df_(response, call='raw'))

            date_init += step_size + datetime.timedelta(days=1)

        if raw_response:
            result = responses
        else:
            result = pd.concat(responses)
            # result = collections.defaultdict(list)
            # for parsed_response in responses:
            #     for key in parsed_response:
            #         result[key].extend(parsed_response[key])

        return result

    def aggregated_site_data(self, r_format='json', api_key=None, system_id=None,
                             start_date=None, end_date=None,
                             aggregate=None, limit_fields=[],
                             user_id=None, raw_response=False):
        """Get aggregated data for given site and date range

        Data includes the following (by default):
            system_id (int)
            measdatetime (datetime)
            availability (float)
            energy_from_array (float)
            poa_irradiation (float)
            energy_to_grid (float)
            energy_from_grid (float)
            total_energy_input (float)
            total_energy_output (float)
            array_energy_fraction (float)
            load_efficiency (float)
            bos_efficiency (float)
            array_yield (float)
            final_yield (float)
            reference_yield (float)
            array_capture_losses (float)
            bos_losses (float)
            system_performance_ratio (float)
            array_performance_ratio (float)
            mean_array_efficiency (float)
            total_system_efficiency (float)

        Note that certain data fields are not provided at all aggregations.
        Try different aggregations if your desired data column is empty.

        Parameters
        ----------
        r_format: str
            format of response
        api_key: str
            PVDAQ API key
        system_id: int
            id of system of interest
        start_date: str
            begin date for data in MM/DD/YYYY format
            if start_date = 'origin', begin date is 01/01/0001
        end_date: str
            end date for data in MM/DD/YYYY format
            if end_date = 'today', current date will be used
        aggregate: str
            frequency for which data is aggregated
            options = (hourly, daily, weekly, monthly)
        limit_fields: list, optional
            specify specific data fields to get (listed above)
            by defualt, all are returned (even if empty)
        user_id: str, optional
            only for admin accounts.
            will return all sites belonging to this id.
        raw_response: bool, optional
            True:
                returns all response of requests.get
            False:
                returns desired aggregate data from requests.get response

        Returns
        -------
        result: dict
            contains response.get response or
            desired aggregate data (detailed above)
        """
        if r_format != 'json':
            raise NotImplementedError('Only json response format is supported at this time.')

        allowed_fields = ['system_id', 'measdatetime', 'availability', 'energy_from_array',
                          'poa_irradiation', 'energy_to_grid', 'energy_from_grid',
                          'total_energy_input', 'total_energy_output', 'array_energy_fraction',
                          'load_efficiency', 'bos_efficiency', 'array_yield', 'final_yield',
                          'reference_yield', 'array_capture_losses', 'bos_losses',
                          'system_performance_ratio', 'array_performance_ratio',
                          'mean_array_efficiency', 'total_system_efficiency']

        req_params = {}

        req_params['api_key'] = self.api_key_check_(api_key)

        req_params['system_id'] = self.system_id_check_(system_id)

        start_date, end_date = self.date_range_check_(start_date, end_date, system_id)
        if any(i is None for i in (start_date, end_date)):
            return None
        req_params['start_date'] = start_date
        req_params['end_date'] = end_date

        if aggregate is None or aggregate not in ('hourly', 'daily', 'weekly', 'monthly'):
            raise ValueError('Invalid aggregate: {}.'.format(aggregate))
        req_params['aggregate'] = aggregate

        if limit_fields:
            if not all(i in allowed_fields for i in limit_fields):
                raise ValueError('Invalid field(s) given in limit_fields: {}.'.
                                 format(','.join([i for i in limit_fields
                                                  if i not in allowed_fields])))
        req_params['limit_fields'] = [x for x in limit_fields]

        if user_id is not None:
            req_params['user_id'] = str(user_id)

        url = self.base_url_ + 'site_data.' + r_format
        response = self.base_api_call_(url, req_params)

        if raw_response:
            result = response
        else:
            result = self.parse_response_outputs_to_df_(response, call='aggregate')

        return result

    def date_range_check_(self, start_date, end_date, system_id):
        """Check start and end date of request for validity.

        If start_date == 'origin', the system metadata will be
        pulled to get earliest available year.
        if end_date == 'today', the system metadata will be pulled to
        get lastest year available (in case data collection stopped).

        Parameters
        ----------
        start_date: str
            earlier date in MM/DD/YYYY format or 'origin'
        end_date: str
            later date in MM/DD/YYYY format or 'today'
        system_id: int
            unique id for system.  needed if 'origin' or 'today'
            passed as start_date or end_date (respectively).

        Returns
        -------
        start_date: str
            earlier date in correct format
        end_date: str
            later date in correct formate
        """
        if start_date is None:
            raise ValueError('Start date must be specified.')
        if end_date is None:
            raise ValueError('End date must be specified.')

        start_date = start_date.lower()
        end_date = end_date.lower()

        if start_date == 'origin' or end_date == 'today':
            metadata = self.sites_metadata(system_id=system_id)

        if start_date == 'origin':
            try:
                year = min(metadata['available_years'])
                start_date = datetime.date(year, 1, 1).strftime('%m/%d/%Y')
            except ValueError as e:
                print('API metadata does not have information on available years.')
                return None, None
        else:
            try:
                _ = datetime.datetime.strptime(start_date, '%m/%d/%Y').date()
            except:
                raise ValueError('Start date {} is not correctly formatted.  Must be MM/DD/YYYY or "origin".'.
                                 format(start_date))

        if end_date == 'today':
            try:
                latest_year = max(metadata['available_years'])
                today = datetime.date.today()# .strftime('%m/%d/%Y')
            except ValueError as e:
                print('API metadata does not have information on available years.')
                return None, None
            if latest_year < today.year:
                end_date = datetime.date(latest_year, 12, 31).strftime('%m/%d/%Y')
            else:
                end_date = datetime.date.today().strftime('%m/%d/%Y')
        else:
            try:
                _ = datetime.datetime.strptime(end_date, '%m/%d/%Y').date()
            except:
                ValueError('End date {} is not correctly formatted.  Must be MM/DD/YYYY or "today".'.
                           format(start_date))

        if datetime.datetime.strptime(start_date, '%m/%d/%Y').date() > \
                datetime.datetime.strptime(end_date, '%m/%d/%Y').date():
            start_date, end_date = end_date, start_date

        return start_date, end_date

    def system_id_check_(self, system_id):
        """Checks if system_id is valid.

        If system_id is invalid, error is raised.  If valid, value is returned as int.

        Parameters
        ----------
        system_id: int/str
            unique system id

        Returns
        -------
        system_id: int/str
        """
        if system_id is None:
            raise ValueError('System ID must be specified.')
        if int(system_id) not in self.allowed_system_ids:
            raise ValueError('System ID {} is invalid.'.format(system_id))
        return system_id

    def api_key_check_(self, api_key):
        """checks if api_key is None

        Return self.api_key_ if api_key is None.  Used to avoid
        passing api_key with every request.

        Parameters
        ----------
        api_key: str
            PVDAQ API key

        Returns
        -------
        api_key: str
        """
        if api_key is None:
            api_key = self.api_key_
        return api_key

    def parse_response_outputs_to_df_(self, response, call=None):
        """Parses json response from API call to dict of lists.

        Parameters
        ----------
        response: dict
            json response from requests call
            assumed to have 'outputs' key which
            is a list of sublists.  first list is the data
            headers.  rest of lists are data values.

        Returns
        -------
        result: pandas DataFrame
            data is stored columnwise
        """
        # columns stored as first element in list
        # data is stored row by row as sublists
        # aggregate outputs field and raw outputs
        # fields slightly differ...
        if call == 'aggregate':
            keys = np.asarray(response['outputs'][0])
            data = np.asarray(response['outputs'][1:])
        elif call == 'raw':
            keys = np.asarray(response['outputs']['data'][0])
            data = np.asarray(response['outputs']['data'][1:])
        else:
            raise ValueError('Call must be either aggregate or raw.')
        # use standard python objects in dict
        # result = {}
        # for key, vec in zip(keys, data.T):
        #     result[key] = list(vec)
        if len(data) == 0:
            result = pd.DataFrame()
        else:
            result = pd.DataFrame(data=data, columns=keys)
        return result

    @staticmethod
    def base_api_call_(url, params):
        """Make request of PVDAQ API.

        Response of API call will contain the following fields:
            outputs (array): array of documents for each site
            errors (arary): array of any errors encountered
            warnings (array): array of warnings encountered
            infos (array): array of non-error and non-warning information
            inputs (dict): list of request parameters
            version (str): web service version

        Parameters
        ----------
        url: str
            url destination for api call
        params: dict
            parameters for api call

        Returns
        -------
        response: dict
            dict of json response
        """
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as error:
            print(error)


if __name__ == '__main__':
    main()
