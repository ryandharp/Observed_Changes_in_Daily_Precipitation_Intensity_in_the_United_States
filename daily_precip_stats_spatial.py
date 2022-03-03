#%% Importing Modules
import pandas as pd
import numpy as np
import time


#%% Specifying Input Values
num_qual_years = 50  # number of years needed for an acceptable station record
percent_qualifying_years = 90  # percentage of years qualifying necessary for an acceptable station record
yearly_pct_threshold = 5  # percent of missing values allowed
precip_min_cutoff = 1  # mm recording needed for a 'rainy' day
first_half_year_min = 1951  # first year for calculating distributions for first half
first_half_year_max = first_half_year_min + 30  # last year for calculating distributions for first half
second_half_year_min = 1991  # first year for calculating distributions for second half
second_half_year_max = second_half_year_min + 30  # last year for calculating distributions for second half
window_length = 11  # number of years for smoothing
window_buffer = int((window_length-1)/2)
threshold = .75  # threshold for percentage of years needed to qualify for a smoothing
percentile_threshold = 0.2  # quantile threshold for whiplash events
zero_year_threshold = 0  # number of zero precip years allowed


#%% Defining Functions
def window_calculations(time_series, window_length, year_list, threshold):
    # running mean calculator
    half_window = window_length // 2
    windowed_data_mean = np.zeros(np.shape(year_list[half_window:-half_window])); windowed_data_mean[:] = np.nan
    windowed_data_std = np.zeros(np.shape(year_list[half_window:-half_window])); windowed_data_std[:] = np.nan
    windowed_data_cov = np.zeros(np.shape(year_list[half_window:-half_window])); windowed_data_cov[:] = np.nan
    c = 0
    for year in year_list[half_window:-half_window]:  # looping through years to use as window centers
        windowed_data = time_series[(time_series.index >= (year - half_window)) & (time_series.index <= (year + half_window))]
        if windowed_data.count() / window_length > threshold:
            windowed_data_mean[c] = windowed_data.mean()
            windowed_data_std[c] = windowed_data.std()
            windowed_data_cov[c] = windowed_data_std[c]/windowed_data_mean[c]
        c += 1
    return windowed_data_mean, windowed_data_std, windowed_data_cov

def load_station(fname):
    # loads data for a given station
    df = pd.read_csv(fname, dtype={'DATE': object, 'PRCP': float, 'TMAX': float, 'TMIN': float, 'PRCP_ATTRIBUTES': str,
                                   'TMAX_ATTRIBUTES': str, 'TMIN_ATTRIBUTES': str}, low_memory=False)
    data = df.filter(['DATE', 'PRCP', 'TMAX', 'TMIN', 'PRCP_ATTRIBUTES', 'TMAX_ATTRIBUTES', 'TMIN_ATTRIBUTES'])
    data['DATE'] = pd.to_datetime(data['DATE'])
    del df
    return data

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

def get_total_days_in_year_series(qual_years):
    # gets the number of days in a year across all qualifying years
    num_days = 0
    for yr in qual_years:
        if yr % 4 == 0:
            days_in_year = 366
        else:
            days_in_year = 365
        num_days = num_days + days_in_year
    return num_days

def get_pct_missing(station_data, days_in_year, total_days, years_array, total_days_series):
    # calculates the amount of missing data in a given station-year and returns relevant summary statistics
    date_mask = (station_data['DATE'].dt.year >= start_year) & (station_data['DATE'].dt.year <= last_year)
    num_days_in_data_year = station_data['PRCP'].groupby(station_data['DATE'][date_mask].dt.year).count()
    num_days_in_data_year_series = pd.Series(num_days_in_data_year, years_array)
    # num_missing_flagged = station_data['PRCP'].isnull().groupby(station_data['DATE'][date_mask].dt.year).sum().astype(int)
    # years_mask = np.isin(years, num_days_in_data_year.index)
    yearly_pct_missing  = 100 - (num_days_in_data_year_series.divide(total_days_series, fill_value=0)*100)
    total_pct_missing = 100 - sum(num_days_in_data_year)/total_days*100
    # yearly_pct_missing = 100 - (days_in_year[years_mask] - num_missing_flagged)/days_in_year[years_mask]*100
    # total_pct_missing = 100 - (days_in_year.sum() - num_missing_flagged.sum())/days_in_year.sum()*100
    return yearly_pct_missing, total_pct_missing


#%% Loading File List
# ghcn_stations = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/US_stations_with_neon.csv')
ghcn_stations = pd.read_csv('/home/rdh0715/US_stations_with_NEON.csv')

qual_stations = ghcn_stations[(ghcn_stations['num_qual_years'] > num_qual_years) & (ghcn_stations['pct_qual_years'] >= percent_qualifying_years)]

qual_station_id = qual_stations['station_id']
qual_lat = qual_stations['latitude']
qual_lon = qual_stations['longitude']
qual_year_length = qual_stations['num_qual_years']
neon_domain = qual_stations['NEON_domain']


#%% Preallocating Arrays
array_size = np.shape(qual_station_id)
station_id_list = []
station_lat = np.zeros(array_size); station_lat[:] = np.nan
station_lon = np.zeros(array_size); station_lon[:] = np.nan
station_early_late = np.zeros(array_size); station_early_late[:] = np.nan
station_early_days = np.zeros(array_size); station_early_days[:] = np.nan
station_late_days = np.zeros(array_size); station_late_days[:] = np.nan
station_both_days = np.zeros(array_size); station_both_days[:] = np.nan
station_domain = np.zeros(array_size); station_domain[:] = np.nan
start_year_array = np.zeros(array_size); start_year_array[:] = np.nan
last_year_array = np.zeros(array_size); last_year_array[:] = np.nan
station_qual_year_length = np.zeros(array_size); station_qual_year_length[:] = np.nan

smoothed_annual_precip_missing_percentage_array = np.zeros(array_size); smoothed_annual_precip_missing_percentage_array[:] = np.nan
smoothed_annual_precip_freq_missing_percentage_array = np.zeros(array_size); smoothed_annual_precip_freq_missing_percentage_array[:] = np.nan
smoothed_mean_dry_missing_percentage_array = np.zeros(array_size); smoothed_mean_dry_missing_percentage_array[:] = np.nan

del array_size


#%% Analysis Prep
qual_flags = ['D', 'G', 'I', 'K', 'L', 'M', 'N', 'O', 'R', 'S', 'T', 'W', 'X', 'Z']  # specifying GHCN flags
j = 0  # starting counter

len_qual_stations = len(qual_station_id)
print(len_qual_stations)

num_regions = len(np.unique(neon_domain))

early_count_total = np.zeros(num_regions); early_count_total[:] = np.nan
late_count_total = np.zeros(num_regions); late_count_total[:] = np.nan
full_count_total = np.zeros(num_regions); full_count_total[:] = np.nan


start = time.time()

for domain in np.unique(neon_domain):  # looping over all domains
    # domain = 5  # using Great Lakes as an example
    domain_stations = qual_stations[neon_domain == domain]  # getting list of all stations in the specified domain

    domain = int(domain)
    print(domain)
    print(domain_stations)
    print(time.time()-start)

    domain_year_array = np.arange(qual_stations['start_year'].min(), qual_stations['last_year'].max(), 1)

    domain_precip_first_half_pdf = np.array([])
    domain_precip_second_half_pdf = np.array([])
    domain_precip_both_first_half_pdf = np.array([])
    domain_precip_both_second_half_pdf = np.array([])

    # initializing early/late/both count variables
    early_count = 0
    late_count = 0
    full_count = 0
    for i, station in domain_stations['station_id'].iteritems():  # looping over all stations in a given domain
        fname = station + '.csv'
        print(fname)

        station_id_list.append(station)
        station_lat[j] = qual_lat.loc[i]
        station_lon[j] = qual_lon.loc[i]
        station_domain[j] = domain
        station_qual_year_length[j] = qual_year_length.loc[i]
        station_early_late[j] = np.nan


        #%% Loading Data
        # fname = 'USW00094728.csv'
        # df = pd.read_csv('~/Documents/great_lakes_precip_variability/raw_data/'+fname)
        df = pd.read_csv('/projects/b1045/GHCN/'+fname, low_memory=False)

        # temp_data = load_station('~/Documents/great_lakes_precip_variability/raw_data/'+fname)
        temp_data = load_station('/projects/b1045/GHCN/'+fname)
        station_data = explode_flags(temp_data)

        start_date = station_data['DATE'].min()
        start_year = start_date.date().year
        start_year_array[j] = start_year
        last_date = station_data['DATE'].max()
        last_year = last_date.date().year
        last_year_array[j] = last_year
        years_array = station_data['DATE'].dt.year.unique()
        years = np.arange(start_year, last_year)

        days_in_year, total_days, total_days_series, total_years_array = get_days_in_year2(years_array, start_year, last_year)
        yearly_pct_missing, total_pct_missing = get_pct_missing(station_data, days_in_year, total_days, years_array, total_days_series)
        qual_years = total_years_array[yearly_pct_missing <= yearly_pct_threshold]

        data = df.filter(['DATE', 'PRCP', 'TMAX', 'TMIN'])
        data['DATE'] = pd.to_datetime(data['DATE'])

        del df, temp_data, station_data, start_date, last_date, years_array
        del days_in_year, total_days, total_days_series, total_years_array, yearly_pct_missing, total_pct_missing


        #%% Data Cleaning
        # converting to mm and removing all events less than 1/3 mm
        data['PRCP'] = data['PRCP']/10  # converting to mm from tenths-mm
        data['PRCP_DATE'] = data['PRCP'] >= precip_min_cutoff
        data['PRCP_QUAL'] = data['PRCP'][data['PRCP_DATE']]
        # year_list = data['DATE'].dt.year.unique()
        full_precip = data['PRCP_QUAL'][data['DATE'].dt.year.isin(qual_years)]
        zero_years = qual_years[full_precip.groupby(data['DATE'].dt.year).sum() == 0]  # testing for weird "valid" records with complete years of zero precipitation
        if len(zero_years) > zero_year_threshold:
            j += 1
            continue

        del zero_years


        #%% Aggregating all observations for the early and late time periods
        first_half_qual_years = qual_years[(qual_years >= first_half_year_min) & (qual_years < first_half_year_max)]
        second_half_qual_years = qual_years[(qual_years >= second_half_year_min) & (qual_years < second_half_year_max)]

        early = False
        late = False
        both = False

        if (start_year <= first_half_year_min) & (last_year >= first_half_year_max):
            first_half_precip = data['PRCP_QUAL'][data['DATE'].dt.year.isin(first_half_qual_years)]
            domain_precip_first_half_pdf = np.concatenate((domain_precip_first_half_pdf, np.array(first_half_precip)))
            station_early_days[j] = get_total_days_in_year_series(first_half_qual_years)
            early_count += 1
            early = True

        if (start_year <= second_half_year_min) & (last_year >= second_half_year_max):
            second_half_precip = data['PRCP_QUAL'][data['DATE'].dt.year.isin(second_half_qual_years)]
            domain_precip_second_half_pdf = np.concatenate((domain_precip_second_half_pdf, np.array(second_half_precip)))
            station_late_days[j] = get_total_days_in_year_series(second_half_qual_years)
            late_count += 1
            late = True

        if (start_year <= first_half_year_min) & (last_year >= second_half_year_max):
            first_half_precip = data['PRCP_QUAL'][data['DATE'].dt.year.isin(first_half_qual_years)]
            domain_precip_both_first_half_pdf = np.concatenate((domain_precip_both_first_half_pdf, np.array(first_half_precip)))
            second_half_precip = data['PRCP_QUAL'][data['DATE'].dt.year.isin(second_half_qual_years)]
            domain_precip_both_second_half_pdf = np.concatenate((domain_precip_both_second_half_pdf, np.array(second_half_precip)))
            station_both_days[j] = station_early_days[j] + station_late_days[j]
            full_count += 1
            both = True

        if both:
            station_early_late[j] = 3
        elif late:
            station_early_late[j] = 2
        elif early:
            station_early_late[j] = 1

        j += 1


#%% Printing Test Results
    np.save('/home/rdh0715/domain_' + str(domain) + '_daily_first_half_pdf.npy', domain_precip_first_half_pdf)
    print(np.shape(domain_precip_first_half_pdf))
    np.save('/home/rdh0715/domain_' + str(domain) + '_daily_second_half_pdf.npy', domain_precip_second_half_pdf)
    print(np.shape(domain_precip_second_half_pdf))
    np.save('/home/rdh0715/domain_' + str(domain) + '_daily_both_first_half_pdf.npy', domain_precip_both_first_half_pdf)
    print(np.shape(domain_precip_both_first_half_pdf))
    np.save('/home/rdh0715/domain_' + str(domain) + '_daily_both_second_half_pdf.npy', domain_precip_both_second_half_pdf)
    print(np.shape(domain_precip_both_second_half_pdf))

    early_count_total[domain - 1] = early_count
    late_count_total[domain - 1] = late_count
    full_count_total[domain - 1] = full_count

results_df = pd.DataFrame(columns = ['station_id', 'station_lat', 'station_lon', 'station_domain', 'early_late'])

results_df['station_id'] = np.array(station_id_list)
results_df['station_lat'] = station_lat
results_df['station_lon'] = station_lon
results_df['station_domain'] = station_domain
results_df['early_late'] = station_early_late
results_df['early_days'] = station_early_days
results_df['late_days'] = station_late_days
results_df['both_days'] = station_both_days


results_df.to_csv('/home/rdh0715/NEON_domain_daily_precip_stats.csv')  # saving summary statistics for each domain




# results_df = pd.DataFrame(columns = ['annual_mean_p', 'annual_mean_slope', 'annual_mean', 'annual_var_p',
#                                      'annual_var_slope', 'annual_var', 'annual_cov_p', 'annual_cov_slope', 'annual_cov',
#                                      'annual_freq_p', 'annual_freq_slope', 'annual_freq', 'per_10_p', 'per_10_slope',
#                                      'per_25_p', 'per_25_slope', 'per_50_p', 'per_50_slope', 'per_75_p', 'per_75_slope',
#                                      'per_90_p', 'per_90_slope', 'per_95_p', 'per_95_slope', 'per_99_p', 'per_99_slope',
#                                      'per_99p5_p', 'per_99p5_slope', 'early_pdf_count', 'late_pdf_count',
#                                      'full_pdf_count'])
#
# results_df['early_pdf_count'] = early_count_total
# results_df['late_pdf_count'] = late_count_total
# results_df['full_pdf_count'] = full_count_total
#
# results_df.to_csv('/home/rdh0715/NEON_domain_daily_precip_stats.csv')


# #%% Populating Pandas Dataframes
# results_df = pd.DataFrame(columns = ['station_id', 'latitude', 'longitude', 'start_year', 'last_year', 'num_qual_years'])
#
# results_smoothed_df = pd.DataFrame(columns = ['station_id', 'latitude', 'longitude', 'start_year', 'last_year', 'num_qual_years'])
#
# results_df['station_id'] = station_id_list
#
# print(np.shape(results_df))
# print(np.shape(station_id_list))
# print(np.shape(station_lon))
# print(j)
#
# results_df['latitude'] = station_lat
# results_df['longitude'] = station_lon
# results_df['start_year'] = start_year_array
# results_df['last_year'] = last_year_array
# results_df['num_qual_years'] = station_qual_year_length
#
# results_df['annual_precip_mean'] = annual_precip_mean_array
#
# results_df['wet_season_start_mon'] = max_precip_mon_array
# results_df['wet_season_mean_total'] = wet_season_mean_array
# results_df['dry_season_start_mon'] = min_precip_mon_array
# results_df['dry_season_mean_total'] = dry_season_mean_array
# results_df['seasonality_index'] = seasonality_index_array
# results_df['wet_to_dry_ratio'] = wet_season_mean_array/dry_season_mean_array
#
# results_df['first_half_wet_season_start_mon'] = first_half_max_precip_mon_array
# results_df['first_half_wet_season_mean_total'] = first_half_wet_season_mean_array
# results_df['first_half_dry_season_start_mon'] = first_half_min_precip_mon_array
# results_df['first_half_dry_season_mean_total'] = first_half_dry_season_mean_array
# results_df['first_half_seasonality_index'] = first_half_seasonality_index_array
#
# results_df['second_half_wet_season_start_mon'] = second_half_max_precip_mon_array
# results_df['second_half_wet_season_mean_total'] = second_half_wet_season_mean_array
# results_df['second_half_dry_season_start_mon'] = second_half_min_precip_mon_array
# results_df['second_half_dry_season_mean_total'] = second_half_dry_season_mean_array
# results_df['second_half_seasonality_index'] = second_half_seasonality_index_array
#
# results_df['ks_test_sig'] = ks_test_p_array
# results_df['ad_test_sig'] = ad_test_p_array
# results_df['first_half_mean'] = first_half_mean_array
# results_df['first_half_std_dev'] = first_half_std_dev_array
# results_df['first_half_skew'] = first_half_skew_array
# results_df['first_half_kurt'] = first_half_kurt_array
# results_df['second_half_mean'] = second_half_mean_array
# results_df['second_half_std_dev'] = second_half_std_dev_array
# results_df['second_half_skew'] = second_half_skew_array
# results_df['second_half_kurt'] = second_half_kurt_array
#
#
# results_smoothed_df['station_id'] = station_id_list
# results_smoothed_df['latitude'] = station_lat
# results_smoothed_df['longitude'] = station_lon
# results_smoothed_df['start_year'] = start_year_array
# results_smoothed_df['last_year'] = last_year_array
# results_smoothed_df['num_qual_years'] = station_qual_year_length
# results_smoothed_df['possible_years'] = possible_years_array
#
# results_smoothed_df['annual_mean_missing_percentage'] = smoothed_annual_precip_missing_percentage_array
#
# results_df.to_csv('/home/rdh0715/precip_stats_' + str(num_qual_years) + '_years_' + str(precip_min_cutoff) + 'mm_threshold_v2.csv')
# results_smoothed_df.to_csv('/home/rdh0715/smoothed_precip_stats_' + str(num_qual_years) + '_years_' + str(precip_min_cutoff) + 'mm_threshold_v2.csv')
#