

import pandas as pd
import os

geological_info = pd.read_json('./cities.json')
geological_info = geological_info.drop(['growth_from_2000_to_2013', 'population'], axis=1)
geological_info['city'] = geological_info['city'].apply(lambda x: x.replace(' ', ''))
geological_info['state'] = geological_info['state'].apply(lambda x: x.replace(' ', ''))

print(geological_info.head())

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

# map = Basemap(projection='merc', lat_0 = 57, lon_0 = -135,
#     resolution = 'h', area_thresh = 0.1,
#     llcrnrlon=-136.25, llcrnrlat=56.0,
#     urcrnrlon=-134.25, urcrnrlat=57.75)

map = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64,
                               urcrnrlat=49, projection='lcc', lat_1=33, lat_2=45,
                               lon_0=-95, resolution='i', area_thresh=10000)

map.drawcoastlines()
map.drawcountries()
map.drawstates()
# map.fillcontinents(color = 'coral')
map.drawmapboundary()

for lon, lat in zip(geological_info['longitude'], geological_info['latitude']):
    x, y = map(lon, lat)
    map.scatter(x, y, marker='x')
    # map.scatter(x, y, s=20)

map.readshapefile(os.path.expanduser('~/Downloads/cb_2016_us_nation_5m/cb_2016_us_nation_5m'), 'us_borders', drawbounds=True)

print(map.us_borders)
# map.plot(geological_info['longitude'], geological_info['latitude'])

plt.show()
