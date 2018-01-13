

import os
import time
import pandas as pd
import numpy as np

import nsrdb_api


def main():
    """Hardcoded API calling script for cities.json file.

    Returns
    -------

    """
    city_data = pd.read_json('./cities.json')

    all_strings = []
    for lat, lon in zip(city_data['latitude'], city_data['longitude']):
        all_strings.append(str(np.round(lon, 4)) + ' ' + str(np.round(lat, 4)))

    names = (1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006,
             2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015)
    attrs = ('ghi', 'clearsky_ghi', 'cloud_type', 'fill_flag')

    responses = []
    for i in range(0, len(all_strings), 100):
        subset = all_strings[i: i + 100]

        caller = nsrdb_api.NSRDBAPI(os.environ.get('NSRDB_API_KEY'), subset, email=os.environ.get('WORK_EMAIL'),
                                    mailing_list=False, affiliation='LBL', full_name='ben', reason='research',
                                    leap_day=False, interval=30, names=names, attributes=attrs)

        response = caller.data_call()
        if response['errors']:
            raise RuntimeError('Error in API call for {} to {}'.format(i, i + 100))
        print(response['outputs']['message'])

        time.sleep(3) # avoid rapid calls - only 1 call every 2 seconds allowed

    print('Done!')


if __name__ == '__main__':
    main()

