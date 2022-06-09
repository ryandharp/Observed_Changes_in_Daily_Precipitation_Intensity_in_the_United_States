import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import pandas as pd

nca_gdf = gpd.read_file('/home/rdh0715/NCARegions/cb_2020_us_state_500k_ncaregions.shp')
nca_gdf['NCA_ID'] = np.arange(1, 11, 1)

region_list = nca_gdf['NCA_ID']
region_shapefiles = list()
for region_id in region_list:
    region_shapefiles.append(nca_gdf[nca_gdf['NCA_ID'] == region_id].unary_union)

def find_which_shapefile(lon, lat, shapefile_list):
    pt = Point([lon, lat])
    for i in np.arange(len(shapefile_list)):
        if pt.within(shapefile_list[i]):
            break
    return i

ghcn_stations = pd.read_csv('/home/rdh0715/ghcn_summary.csv')
US_stations = ghcn_stations[ghcn_stations['station_id'].str.slice(0,2) == 'US']

nca_region_ind = np.zeros(np.shape(US_stations)[0]); nca_region_ind[:] = np.nan
nca_region = np.zeros(np.shape(US_stations)[0]); nca_region[:] = np.nan

c = 0
for i, station in US_stations.iterrows():
# for i, station in test.iterrows():
    nca_region_ind[c] = find_which_shapefile(station['longitude'], station['latitude'], region_shapefiles)
    c += 1

for ind, region in np.ndenumerate(nca_region_ind):
    nca_region[ind] = int(region_list.values[int(region)])

US_stations['NCA_region'] = nca_region
US_stations.to_csv('/home/rdh0715/US_stations_with_NCA.csv')
