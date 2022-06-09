#%% Pulling Data from GHCN Daily Summary Files
# load modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time


#%% Loading Individual Station Data
def load_station(fname):
    # this loads in station data for an individual station
    df = pd.read_csv(fname, dtype={'DATE': object, 'PRCP': float, 'TMAX': float, 'TMIN': float, 'PRCP_ATTRIBUTES': str,
                                   'TMAX_ATTRIBUTES': str, 'TMIN_ATTRIBUTES': str}, low_memory=False)
    data = df.filter(['DATE', 'PRCP', 'TMAX', 'TMIN', 'PRCP_ATTRIBUTES', 'TMAX_ATTRIBUTES', 'TMIN_ATTRIBUTES'])
    data['DATE'] = pd.to_datetime(data['DATE'])
    del df
    return data


#%% Exploding Flags and NaN-ing Flagged Values
def explode_flags(compact_data):
    # this expands flagged data and automatically reassigns flagged values as nan
    min_ind = compact_data.index[compact_data['PRCP_ATTRIBUTES'].notnull()].min()
    if compact_data['PRCP_ATTRIBUTES'][min_ind].count(',') == 3:
        compact_data[['PRCP_MEAS_FLAG', 'PRCP_QUAL_FLAG', 'PRCP_SOURCE_CODE', 'PRCP_TIME']] = compact_data['PRCP_ATTRIBUTES'].str.split(',', expand=True)
    elif compact_data['PRCP_ATTRIBUTES'][min_ind].count(',') == 2:
        compact_data[['PRCP_MEAS_FLAG', 'PRCP_QUAL_FLAG', 'PRCP_SOURCE_CODE']] = compact_data['PRCP_ATTRIBUTES'].str.split(',', expand=True)
    flag_mask = (compact_data['PRCP_QUAL_FLAG'].isna()) | (compact_data['PRCP_QUAL_FLAG'].isin(qual_flags))
    compact_data['PRCP'] = compact_data['PRCP'].where(~flag_mask, np.nan)
    return compact_data


#%% Creating Year Arrays
def get_days_in_year2(years_array, first_year, last_year):
    # gets the number of days in a year
    days_in_year_array = np.zeros(np.shape(years_array))
    i = 0
    for yr in years_array:
        if yr % 4 == 0:
            days_in_year_array[i] = 366
        else:
            days_in_year_array[i] = 365
        i += 1
    total_years_array = np.arange(first_year, last_year+1)
    total_days_array = np.zeros(np.shape(total_years_array))
    i = 0
    for yr in total_years_array:
        if yr % 4 == 0:
            total_days_array[i] = 366
        else:
            total_days_array[i] = 365
        i += 1
    total_days = sum(total_days_array)
    total_days_series = pd.Series(total_days_array, index=total_years_array)
    return days_in_year_array, total_days, total_days_series, total_years_array


#%% Checking for missing data
def get_pct_missing(station_data, days_in_year, total_days, years_array, total_days_series):
    # calculates the amount of missing data in a given station-year and returns relevant summary statistics
    date_mask = (station_data['DATE'].dt.year >= start_year) & (station_data['DATE'].dt.year <= last_year)
    num_days_in_data_year = station_data['PRCP'].groupby(station_data['DATE'][date_mask].dt.year).count()
    num_days_in_data_year_series = pd.Series(num_days_in_data_year, years_array)
    yearly_pct_missing  = 100 - (num_days_in_data_year_series.divide(total_days_series, fill_value=0)*100)
    total_pct_missing = 100 - sum(num_days_in_data_year)/total_days*100
    return yearly_pct_missing, total_pct_missing


#%% Loading Station Metadata File
width = [
    11, #station_id
    9, #latitude
    10, #longitude
    7, #elevation
    3, #state
    31, #name
    4, #gsn_id
    4, #hcn_crn_id
    6] #wmo_id

df = pd.read_fwf('/projects/b1045/GHCN/ghcnd-stations.txt',
                 widths=width, names=['station_id', 'lat', 'lon', 'elev', 'state', 'name', 'gsn_id', 'hcn_crn_id', 'wmo_id'])
stations_meta = df
del df

# df = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/ghcnd-stations_file_compare.csv')
df = pd.read_csv('/projects/b1045/GHCN/ghcnd-stations_file_compare.csv')
unavailable_files = df['missing_from_available']
del df


#%% Preallocating variables
station_id = np.empty(np.shape(stations_meta)[0], dtype=object); station_id[:] = np.nan
latitude = np.zeros(np.shape(stations_meta)[0]); latitude[:] = np.nan
longitude = np.zeros(np.shape(stations_meta)[0]); longitude[:] = np.nan
elevation = np.zeros(np.shape(stations_meta)[0]); elevation[:] = np.nan
name = np.zeros(np.shape(stations_meta)[0], dtype=object); name[:] = np.nan

start_year_array = np.zeros(np.shape(stations_meta)[0]); start_year_array[:] = np.nan
last_year_array = np.zeros(np.shape(stations_meta)[0]); last_year_array[:] = np.nan
num_years_qual_array = np.zeros(np.shape(stations_meta)[0]); num_years_qual_array[:] = np.nan
pct_years_qual_array = np.zeros(np.shape(stations_meta)[0]); pct_years_qual_array[:] = np.nan
total_pct_missing_array = np.zeros(np.shape(stations_meta)[0]); pct_years_qual_array[:] = np.nan
test_array = np.zeros([np.shape(stations_meta)[0], 2]); test_array[:] = np.nan
qual_years_array = list([])
annual_pct_missing_array = list([])
ind_array = list([])


#%% Looping over all stations
qual_flags = ['D', 'G', 'I', 'K', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'W', 'X', 'Z']
yearly_pct_threshold = 10  # percentage of missing days allowed in a qualifying year
start_time = time.time()

for ind, row in stations_meta[~stations_meta['station_id'].isin(unavailable_files)].iterrows():
    print(ind)
    fname = '/projects/b1045/GHCN/' + row.station_id + '.csv'
    temp_data = load_station(fname)
    if ("PRCP" in temp_data) is False:
        print('skipping: no precip data')
        continue
    elif ("PRCP_ATTRIBUTES" in temp_data) is False:
        print('skipping: no precip data')
        continue
    station_data = explode_flags(temp_data)

    # getting first/last date
    start_date = station_data['DATE'].min()
    start_year = start_date.date().year
    last_date = station_data['DATE'].max()
    last_year = last_date.date().year
    start_year_array[ind] = start_year
    last_year_array[ind] = last_year

    # getting year arrays
    years_array = station_data['DATE'].dt.year.unique()
    days_in_year, total_days, total_days_series, total_years_array = get_days_in_year2(years_array, start_year, last_year)
    yearly_pct_missing, total_pct_missing = get_pct_missing(station_data, days_in_year, total_days, years_array, total_days_series)
    qual_years = total_years_array[yearly_pct_missing <= yearly_pct_threshold]
    num_qual_years = len(qual_years)
    pct_years_qual = num_qual_years/len(total_years_array)*100
    num_years_qual_array[ind] = num_qual_years
    pct_years_qual_array[ind] = pct_years_qual
    total_pct_missing_array[ind] = total_pct_missing
    qual_years_array.append(qual_years)
    annual_pct_missing_array.append(yearly_pct_missing)
    station_id[ind] = row.station_id
    latitude[ind] = row.lat
    longitude[ind] = row.lon
    elevation[ind] = row.elev
    name[ind] = row['name']
    ind_array.append(ind)

end_time = time.time()
print(end_time-start_time)

len_years = last_year_array - start_year_array


#%% Creating metadata file
# creating pandas dataframe
ghcn_summary = pd.DataFrame({'station_id': [], 'latitude': [], 'longitude': [], 'elevation': [], 'name': [],
                              'start_year': [], 'last_year': [], 'len_years': [], 'num_qual_years': [],
                              'pct_qual_years': [], 'pct_days_missing': []})
# populating dataframe
ghcn_summary['station_id'] = station_id
ghcn_summary['latitude'] = latitude
ghcn_summary['longitude'] = longitude
ghcn_summary['elevation'] = elevation
ghcn_summary['name'] = name
ghcn_summary['start_year'] = start_year_array
ghcn_summary['last_year'] = last_year_array
ghcn_summary['len_years'] = len_years
ghcn_summary['num_qual_years'] = num_years_qual_array
ghcn_summary['pct_qual_years'] = pct_years_qual_array
ghcn_summary['pct_days_missing'] = total_pct_missing_array

ghcn_summary = ghcn_summary[~ghcn_summary['station_id'].isnull()]
ghcn_summary.to_csv('/home/rdh0715/ghcn_summary.csv')  # saving metadata/summary file
