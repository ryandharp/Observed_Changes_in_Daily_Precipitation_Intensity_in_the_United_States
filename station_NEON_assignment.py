import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd

neon_gdf = gpd.read_file('/home/rdh0715/NEONDomains_0/NEON_Domains.shp')

domain_list = neon_gdf['DomainID']
domain_shapefiles = list()
for domain_id in domain_list:
    domain_shapefiles.append(neon_gdf[neon_gdf['DomainID'] == domain_id].unary_union)

def find_which_shapefile(lon, lat, shapefile_list):
    pt = Point([lon, lat])
    for i in np.arange(len(shapefile_list)):
        if pt.within(shapefile_list[i]):
            break
    return i

ghcn_stations = pd.read_csv('/home/rdh0715/ghcn_summary.csv')
US_stations = ghcn_stations[ghcn_stations['station_id'].str.slice(0,2) == 'US']
# test = US_stations[(US_stations['num_qual_years'] > 50) & (US_stations['pct_qual_years'] >= 90)]

neon_domain_ind = np.zeros(np.shape(US_stations)[0]); neon_domain_ind[:] = np.nan
neon_domain = np.zeros(np.shape(US_stations)[0]); neon_domain[:] = np.nan

c = 0
for i, station in US_stations.iterrows():
# for i, station in test.iterrows():
    neon_domain_ind[c] = find_which_shapefile(station['longitude'], station['latitude'], domain_shapefiles)
    c += 1

for ind, domain in np.ndenumerate(neon_domain_ind):
    neon_domain[ind] = int(domain_list.values[int(domain)])

US_stations['NEON_domain'] = neon_domain
US_stations.to_csv('/home/rdh0715/US_stations_with_NEON.csv')
