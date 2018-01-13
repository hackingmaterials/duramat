
import requests
import os


INTERVALS = (30, 60)
NAMES = (1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 'tmy')
ATTRIBUTES = ('dhi', 'dni', 'ghi', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'cloud_type',
              'dew_point', 'surface_air_temperature_nwp', 'surface_pressure_background',
              'surface_relative_humidity_nwp', 'solar_zenith_angle', 'total_precipitable_water_nwp',
              'wind_direction_10m_nwp', 'wind_speed_10m_nwp', 'fill_flag')
MAX_WEIGHT = 175000000


def main():
    """Test NSRDB calls (site count and data download).

    These example parameters are taken from the NSRDB developer pages.
    """
    caller = NSRDBAPI(os.environ.get('NSRDB_API_KEY'), ('-106.22 32.9741', '-106.18 32.9741', '-106.1 32.9741'),
                      email='bhellis@lbl.gov', interval=60, leap_day=False,
                      utc=False, affiliation='LBL', full_name='Honored User', mailing_list=False, reason='Academic',
                      names=(2012, 2013), attributes=('ghi', 'clearsky_ghi', 'cloud_type'))
    print(caller.site_count_call())
    print(caller.data_call())


class NSRDBAPI(object):
    """
    Call NSRDB API to get PSM data.

    NSRDB provides two options for downloading data - direct CSV download and by email delivery.  The CSV option
    limits each call to a single location and a single year.  This wrapper does *not* currently support this strategy.
    This wrapper supports the email delivery option since it can provide data for multiple locations and years
    in a single API call.  This method requires having a Globus account to actually retrieve the data (in my experience
    an email is sent telling the user to download data from Globus).

    More information on the API can be found at https://developer.nrel.gov/docs/solar/nsrdb/
    """

    def __init__(self, api_key, wkt, email=None, mailing_list=False, affiliation=None, full_name='user1',
                 reason=None, leap_day=False, utc=False, interval=30,
                 names=(1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                        2013, 2014, 2015, 'tmy'),
                 attributes=('dhi', 'dni', 'ghi', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'cloud_type',
                             'dew_point', 'surface_air_temperature_nwp', 'surface_pressure_background',
                             'surface_relative_humidity_nwp', 'solar_zenith_angle', 'total_precipitable_water_nwp',
                             'wind_direction_10m_nwp', 'wind_speed_10m_nwp', 'fill_flag')):
        """Set object parameters.

        Parameters
        ----------
        api_key: str
            NSRDB API key.
        wkt: list-like
            Well-known text of longitude/latitude points.  Should be a list/tuple of points.
            Longitude and latitude for a single point must be strings separated by spaces.  Order matters - it must be
            'longitude latitude'.
        email: str
            Address where data will be delivered.  This is required data_call method.  The site_count method will
            work without an email.
        mailing_list: bool
            Sign up for mailing list.
        affiliation: str
            Who you work for/where you work.
        full_name: str
            Your name.
        reason: str
            Why are you acquiring data?
        leap_day: bool
            Include leap day in downloaded data
        utc: bool
            Convert local time to UTC?
        interval: int
            Data frequency, either 30 or 60 minutes.
        names: list-like
            Years of interest for data.
        attributes: list-like
            String values of desired measurements.
        """
        self.api_key = api_key
        self.wkt = wkt
        self.email = email
        self.mailing_list = str(mailing_list).lower()
        self.affiliation = affiliation
        self.full_name = full_name
        self.reason = str(reason).lower()
        self.leap_day = str(leap_day).lower()
        self.utc = str(utc).lower()

        if interval not in INTERVALS:
            raise ValueError('Interval must be in {}.'.format(INTERVALS))
        else:
            self.interval = interval

        if any(i not in NAMES for i in names):
            raise ValueError('Parameter names has invalid argument(s).  May be any of the following: {}'.format(NAMES))
        else:
            self.names = [str(i) for i in names]

        if any(i not in ATTRIBUTES for i in attributes):
            raise ValueError('Parameter attributes has invalid argument(s).  '
                             'May be any of the following: {}'.format(ATTRIBUTES))
        else:
            self.attributes = attributes

    def calc_call_weight(self, estimate_site_count=True):
        """Calculates 'weight' of an API call.  Maximum weight of a call is 175000000.

        Formula (per https://developer.nrel.gov/docs/solar/nsrdb/guide):
        weight = site-count * attribute-count * year-count * data-intervals-per-year
            - site-count is derived from the WKT value submitted and can be retrieved using the site_count API endpoint.
            - attribute-count is equal to the number of attributes requested
            - year-count is equal to the number of years requested
            - data-intervals-per-year is ((60/interval)*24*365) where interval is the interval requested

        The site-count is esimated as len(self.wkt) (assumed to be POINT/MULTIPOINT).
        The handling of wkt can be refactored to include other wkt formats like POLYGON, but then the site_count API
        call will have to be used.

        Parameters
        ----------
        estimate_site_count: bool
            Estimate the number of sites (if True) or use site_count API call.

        Returns
        -------
        weight: int
            Weight of call.
        """
        if estimate_site_count:
            site_count = len(self.wkt)
        else:
            # assuming that nsrdb_site_count from response outputs is the value used
            # there are several other outputs given, not sure which is actually correct
            site_count = self.site_count_call()['outputs']['nsrdb_site_count']
        weight = len(self.wkt) * len(self.attributes) * len(self.names) * ((60 / self.interval) * 24 * 365)
        return weight

    def _make_url(self, mode):
        """Make url for requests call.

        Parameters
        Parameters
        ----------
        mode: str
            Must be 'data' or 'site_count' to specify which call is desired.

        Returns
        -------
        url: str
            URL needed for API call.
        """
        if mode.lower().strip() == 'data':
            url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.json?api_key=' + self.api_key
        elif mode.lower().strip() == 'site_count':
            url = 'http://developer.nrel.gov/api/solar/nsrdb/site_count.json?api_key=' + self.api_key
        else:
            raise ValueError('Invalid option for mode parameter.')
        return url

    def _make_wkt(self):
        """Format wkt points for API call.

        Returns
        -------
        wkt: str
            Formatted wkt for call.
        """
        wkt = ','.join(self.wkt)
        if len(self.wkt) > 1:
            wkt = 'MULTIPOINT(' + wkt + ')'
        else:
            wkt = 'POINT(' + wkt + ')'
        return wkt

    def _make_payload(self):
        """Make payload dictionary of API call parameters.

        Returns
        -------
        payload: dict
            Expected keys/values for API call.
        """
        wkt = self._make_wkt()
        payload = {'names': ','.join(self.names),
                   'leap_day': self.leap_day,
                   'interval': self.interval,
                   'full_name': self.full_name,
                   'email': self.email,
                   'affiliation': self.affiliation,
                   'mailing_list': self.mailing_list,
                   'reason': self.reason,
                   'attributes': ','.join(self.attributes),
                   'wkt': wkt}
        return payload

    def data_call(self, estimate_site_count=False):
        """Call NSRDB to get PSM data.

        The data is not explicitly returned.  Instead, it will be sent as a .zip file to the email supplied.

        Parameters
        ----------
        estimate_site_count: bool
            Use an estimate or an exact count for site count in the weight calculation.

        Returns
        -------
        response.json(): dict
            Dictionary (from JSON) of response from API call.
        """
        if self.email is None:
            raise ValueError('You must supply an email')
        call_weight = self.calc_call_weight(estimate_site_count=estimate_site_count)
        if call_weight > MAX_WEIGHT:
            raise RuntimeError('API call is over maximum weight ({} > {}).'.format(call_weight, MAX_WEIGHT))
        # else:
        #     print('Call weight acceptable ({} < {})'.format(call_weight, MAX_WEIGHT))
        url = self._make_url('data')
        payload = self._make_payload()
        headers = {'content-type': 'application/x-www-form-urlencoded', 'cache-control': 'no-cache'}
        response = requests.request('POST', url, data=payload, headers=headers)
        return response.json()

    def site_count_call(self):
        """Call NSRDB to get number of sites in call.

        Returns
        -------
        response.json(): dict
            Dictionary (from JSON) of response from API call.
        """
        url = self._make_url('site_count')
        payload = {'wkt': self._make_wkt()}
        response = requests.get(url, params=payload)
        return response.json()


if __name__ == '__main__':
    main()

