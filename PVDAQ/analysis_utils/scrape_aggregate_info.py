

import pvdaq_api
import sys
import pandas as pd
import datetime
import collections
import argparse

# allowed time periods for when data is aggregated
PERIODS = ('hourly', 'daily', 'weekly', 'monthly')

# allowed system_ids (gathered by hand from PVDAQ site - check back for more)
SITES = [2, 3, 4, 10, 17, 18, 33, 34, 35, 38,
         39, 50, 51, 1199, 1207, 1208, 1229,
         1230, 1231, 1236, 1239, 1276, 1277,
         1278, 1283, 1284, 1289, 1292, 1332,
         1389, 1403, 1405, 1422, 1423, 1424,
         1425, 1429]


def main(argv):
    description = 'Script to collect aggregate data from PVDAQ website.'
    parser = argparse.ArgumentParser(description=description)
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-k', '--key', default=None, required=True, help='PVDAQ API key needed to access site data.')
    required_named.add_argument('-p', '--period', default=None, required=True,
                                help='Time period of the aggregation for which to get data. Options are {}'.format(' '.join(PERIODS)))
    parser.add_argument('-sid', '--system_id', default=None, required=False, nargs='+',
                        help='Specific system_id(s) separated by spaces. Options are {}'.format(' '.join([str(x) for x in SITES])))

    args = parser.parse_args(argv)
    key = args.key
    period = args.period
    if period not in PERIODS:
        raise ValueError('period must be one of the following: {}'.format(','.join(PERIODS)))
    if args.system_id is not None:
        sites = [int(x) for x in args.system_id]
    else:
        sites = SITES
    scrape_aggregate_data(key, period, sites)


def scrape_aggregate_data(key, period, sites):
    '''
    gather aggregated site data from PVDAQ for a given period of aggregation
    will print any sites to a CSV in the current directory.

    each site is given up to 2 chances for a successful call.  script will exit if
    failing sites reach 2 attemps.

    TODO: add functionality so users can additional API options that are currently not offered

    arguments:
        key (str)
            API key for accessing PVDAQ site
        period (str)
            period for which aggregation occurred
        sites (list-like of ints)
            list of system_ids to gather data from

    returns:
        None
    '''
    api = pvdaq_api.PVDAQ_API(key)
    failed_counts = collections.defaultdict(int)

    while sites:
        site = sites.pop(0)
        print('{}\r'.format(site))
        try:
            info = api.sites({'system_id': site})
        except:
            sites.append(site)
            continue

        min_yr = min(info['available_years'])
        min_date = datetime.date(min_yr, 1, 1)
        max_yr = max(info['available_years'])
        if max_yr == datetime.date.today().year:
            max_date = datetime.date.today()
        else:
            max_date = datetime.date(max_yr, 12, 31)
        min_date = min_date.strftime('%m/%d/%Y')
        max_date = max_date.strftime('%m/%d/%Y')

        try:
            agg_data = api.aggregate({'system_id': site,
                                      'start_date': min_date,
                                      'end_date': max_date,
                                      'aggregate': period})
            df = pd.DataFrame.from_dict(agg_data)
            df.to_csv('./{}-{}.csv'.format(site, period), index=False)
            if site in failed_counts:
                del failed_counts[site]
        except:
            failed_counts[site] += 1
            if all(i >= 2 for i in failed_counts.values()):
                print('error processing sites {}'.format(failed_counts.keys()))
                # sys.exit(-1)
                return failed_counts.keys()
            sites.append(site)
            continue
        return faild_counts.keys()

if __name__ == '__main__':
    main(sys.argv[1:])
