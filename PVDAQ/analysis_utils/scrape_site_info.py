'''
script for collecting site metadata from PVDAQ
'''


import sys
import argparse
from collections import namedtuple
import pvdaq_api
import pandas as pd

# constants for system_ids and geographic/climate data gathered by hand
# site Koppen-Geiger climate class gotten from
# http://koeppen-geiger.vu-wien.ac.at/present.htm
# with use of google earth
climate_location = namedtuple('climate_location', ['climate', 'metro_area', 'state'])

# keys are allowed system_ids (gotten by hand - check back for additional sites)
# this list needs to be updated - skipping the climate, metro_area, and state for now
# SITES = {2:    climate_location('BSk', 'Denver', 'CO'),              3: climate_location('BSk', 'Denver', 'CO'),
#          4:    climate_location('BSk', 'Denver', 'CO'),             10: climate_location('BSk', 'Denver', 'CO'),
#          17:   climate_location('BSk', 'Denver', 'CO'),             18: climate_location('BSk', 'Denver', 'CO'),
#          33:   climate_location('BSk', 'Denver', 'CO'),             34: climate_location('BSk', 'Denver', 'CO'),
#          35:   climate_location('BSk', 'Denver', 'CO'),             38: climate_location('BSk', 'Denver', 'CO'),
#          39:   climate_location('BSk', 'Denver', 'CO'),             50: climate_location('BSk', 'Denver', 'CO'),
#          51:   climate_location('BSk', 'Denver', 'CO'),           1199: climate_location('Cfa', 'Baltimore', 'MD'),
#          1207: climate_location('BSk', 'Salt Lake City', 'UT'),   1208: climate_location('BSk', 'Denver', 'CO'),
#          1229: climate_location('Cfa', 'Lakeland', 'FL'),         1230: climate_location('Csa', 'Livermore', 'CA'),
#          1231: climate_location('Cfa', 'Daytona Beach', 'FL'),    1236: climate_location('Cfa', 'Trenton', 'NJ'),
#          1239: climate_location('Dfb', 'Presque Isle', 'ME'),     1276: climate_location('BWh', 'Las Vegas', 'NV'),
#          1277: climate_location('BWh', 'Las Vegas', 'NV'),        1278: climate_location('BWh', 'Las Vegas', 'NV'),
#          1283: climate_location('BSk', 'Denver', 'CO'),           1284: climate_location('BSk', 'Denver', 'CO'),
#          1289: climate_location('BSk', 'Denver', 'CO'),           1292: climate_location('BSk', 'Golden', 'CO'),
#          1332: climate_location('BSk', 'Golden', 'CO'),           1389: climate_location('Cfa', 'Titusville', 'FL'),
#          1403: climate_location('Cfa', 'Titusville', 'FL'),       1405: climate_location('BSk', 'Albuquerque', 'NM'),
#          1422: climate_location('Dfb', 'Burlington', 'VT'),       1423: climate_location('BWh', 'Henderson', 'NV'),
#          1424: climate_location('Dfb', 'Burlington', 'VT'),       1425: climate_location('BWh', 'Henderson', 'NV'),
#          1429: climate_location('BSk', 'Albuquerque', 'NM')}


def main(argv):
    parser = argparse.ArgumentParser(description='Script to collect all site metadata from PVDAQ website.')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-k', '--key', default=None, required=True, help='PVDAQ API key needed to access site data.')
    args = parser.parse_args(argv)
    key = args.key
    api = pvdaq_api.PVDAQ_API(key)
    scrape_site_data(api)


def scrape_site_data(api):
    '''Gather site data metadata from PVDAQ.

    Data will be printed as CSV file in current directory.
    Metadata is augmented with climate class, 'nearest' city, and state (all gathered by hand).
    Site data is printed to current working directory as site-info.csv

    Parameters
    ----------
    api: object
        PVDAQ API object

    Returns
    -------
    None
    '''
    df_list = []
    for site in api.allowed_system_ids:
        info = api.sites_metadata(**{'system_id': site})
        if len(info['available_years']) > 0:
            info['min_year'] = min([int(x) for x in info['available_years']])
            info['max_year'] = max([int(x) for x in info['available_years']])
        else:
            info['min_year'] = None
            info['max_year'] = None
        # info['climate'] = SITES[site].climate
        # info['metro_area'] = SITES[site].metro_area
        # info['state'] = SITES[site].state
        # info['country'] = 'USA'
        del info['available_years']
        df_list.append(pd.DataFrame(info))
    master_df = pd.concat(df_list, axis=1).T
    master_df.to_csv('./site-info.csv')


if __name__ == '__main__':
    main(sys.argv[1:])
