
import requests
import os
import warnings

INTERVALS = (30, 60)
NAMES = (1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
         2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 'tmy')
ATTRIBUTES = ('dhi', 'dni', 'ghi', 'clearsky_dhi', 'clearsky_dni', 'clearsky_ghi', 'cloud_type',
              'dew_point', 'surface_air_temperature_nwp', 'surface_pressure_background',
              'surface_relative_humidity_nwp', 'solar_zenith_angle', 'total_precipitable_water_nwp',
              'wind_direction_10m_nwp', 'wind_speed_10m_nwp', 'fill_flag')
MAX_WEIGHT = 175000000

class NSRDBCaller(object):
    """
    weight = site-count*attribute-count*year-count*data-intervals-per-year
    site-count is derived from the WKT value submitted and can be retrieved using the site_count API endpoint.
    attribute-count is equal to the number of attributes requested
    year-count is equal to the number of years requested
    data-intervals-per-year is ((60/interval)*24*365) where interval is the interval requested
    """

    def __init__(self, email, api_key, wkt, mailing_list=False, affiliation=None, full_name='user1',
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
        email: str
            Address where data will be delivered.
        api_key: str
            NSRDB API key (see https://developer.nrel.gov/docs).
        wkt: list-like
            Well-known text of latitute/longitude points.  Should be a list/tuple of latitude and longitude points.
            Latitude and longitude for a single point must be strings separated by spaces.
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
        self.email = email
        self.api_key = api_key
        self.wkt = wkt
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

    def _calc_call_weight(self):
        """Calculates 'weight' of an API call.  Maximum allowed weight of a call is 175000000.

        Formula (from https://developer.nrel.gov/docs/solar/nsrdb/guide):
        weight = site-count*attribute-count*year-count*data-intervals-per-year
            - site-count is derived from the WKT value submitted and can be retrieved using the site_count API endpoint.
            - attribute-count is equal to the number of attributes requested
            - year-count is equal to the number of years requested
            - data-intervals-per-year is ((60/interval)*24*365) where interval is the interval requested
        The site-count will be esimated as len(self.wkt)
        """
        weight = len(self.wkt) * len(self.attributes) * len(self.names) * ((60 / self.interval) * 24 * 365)
        return weight

    def _make_url(self):
        """Make url for requests call.

        Returns
        -------
        url: str
            URL needed for API call.
        """
        url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.json?api_key=' + self.api_key
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

    def call(self):
        print(self._calc_call_weight())
        if self._calc_call_weight() > MAX_WEIGHT:
            raise RuntimeError('API call is over maximum weight.')
        url = self._make_url()
        payload = self._make_payload()
        headers = {'content-type': 'application/x-www-form-urlencoded', 'cache-control': 'no-cache'}
        response = requests.request("POST", url, data=payload, headers=headers)
        return response.text

caller = NSRDBCaller('bhellis@lbl.gov', os.environ.get('NSRDB_API_KEY'),
                     ('-106.22 32.9741', '-106.18 32.9741', '-106.1 32.9741'), interval=60, leap_day=False, utc=False,
                     affiliation='LBL', full_name='Honored User', mailing_list=False, reason='Academic',
                     names=(2012, 2013), attributes=('ghi', 'clearsky_ghi', 'cloud_type'))

print(caller.call())


# key = os.environ.get('NSRDB_API_KEY')
# if key is None:
#     raise ValueError('Bad key')
#
# url = "http://developer.nrel.gov/api/solar/nsrdb_0512_download.json?api_key=" + key
#
# payload = "names=2012,2013&leap_day=false&interval=60&utc=false&full_name=Honored%2BUser&email=bhellis@lbl.gov&affiliation=LBL&mailing_list=true&reason=Academic&attributes=dhi%2Cdni%2Cwind_speed_10m_nwp%2Csurface_air_temperature_nwp&wkt=MULTIPOINT(-106.22%2032.9741%2C-106.18%2032.9741%2C-106.1%2032.9741)"
#
# headers = {
#     'content-type': "application/x-www-form-urlencoded",
#     'cache-control': "no-cache"
# }
#
# response = requests.request("POST", url, data=payload, headers=headers)
#
# print(response.text)
