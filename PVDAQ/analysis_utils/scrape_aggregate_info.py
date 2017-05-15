

import sys
import collections
import copy
import argparse
import pvdaq_api


# formats for output
FORMAT = ('csv', 'json')

# allowed time periods for when data is aggregated
PERIODS = ('hourly', 'daily', 'weekly', 'monthly')

# allowed system_ids (gathered by hand from PVDAQ site - check back for more)
# SITES = [2, 3, 4, 10, 17, 18, 33, 34, 35, 38,
#          39, 50, 51, 1199, 1207, 1208, 1229,
#          1230, 1231, 1236, 1239, 1276, 1277,
#          1278, 1283, 1284, 1289, 1292, 1332,
#          1389, 1403, 1405, 1422, 1423, 1424,
#          1425, 1429]


def make_parser():
    description = 'Script to collect aggregate data from PVDAQ website.'
    parser = argparse.ArgumentParser(description=description)
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-k', '--key', default=None, required=True,
                                help='PVDAQ API key needed to access site data.')
    required_named.add_argument('-p', '--period', default=None, required=True,
                                help='Time period of the aggregation for which to get data. Options are {}.'.format(','.join(PERIODS)))
    parser.add_argument('-f', '--format', default='csv', required=False,
                        help='Format of saved file.  Options are {}.'.format(','.join(FORMAT)))
    parser.add_argument('-sid', '--system_id', default=None, required=False, nargs='*',
                        help='Specific system_id(s) separated by spaces.')
    return parser


def main(argv):
    parser = make_parser()
    args = parser.parse_args(argv)
    key = args.key
    period = args.period
    fileformat = args.format
    sites = args.system_id
    if period not in PERIODS:
        raise ValueError('period must be one of the following: {}'.format(','.join(PERIODS)))
    if fileformat not in FORMAT:
        raise ValueError('format must be one of the following: {}'.format(','.join(FORMAT)))
    if args.system_id is not None:
        sites = [int(x) for x in args.system_id]
    api = pvdaq_api.PVDAQ_API(key)
    scrape_aggregate_data(api, period, sites, fileformat)


def scrape_aggregate_data(api, period, sites, fileformat):
    '''Get aggregated site data.

    Gather aggregated site data from PVDAQ for a given period of aggregation
    will print any sites to a CSV in the current directory.

    Each site is given up to 2 chances for a successful call.  Data is printed
    to current directory using following convention:
        {site}-{period}.{fileformat}

    Parameters
    ----------
    api: object
        PVDAQ_API object
    period: str
        period for which aggregation occurred
    sites: list-like of ints
        list of system_ids to gather data from

    Returns
    -------
    None
    '''
    failed_counts = collections.defaultdict(int)
    if sites is None:
        sites = [x for x in api.allowed_system_ids]

    while sites:
        site = sites.pop(0)

        try:
            agg_data = api.aggregated_site_data(**{'system_id': site,
                                                   'start_date': 'origin',
                                                   'end_date': 'today',
                                                   'aggregate': period})
            if agg_data is None:
                continue
            if fileformat == 'csv':
                agg_data.to_csv('./{}-{}.csv'.format(site, period), index=False)
            elif fileformat == 'json':
                agg_data.to_json('./{}-{}.json'.format(site, period), orient='index')
            else:
                raise ValueError('fileformat unrecognized.')
            if site in failed_counts:
                del failed_counts[site]
        except:
            print('error with site {}'.format(site))
            failed_counts[site] += 1
            if all(i > 2 for i in failed_counts.values()):
                raise RuntimeError('Repeated site failures for sites {}.'.format(failed_counts))
            sites.append(site)


if __name__ == '__main__':
    main(sys.argv[1:])
