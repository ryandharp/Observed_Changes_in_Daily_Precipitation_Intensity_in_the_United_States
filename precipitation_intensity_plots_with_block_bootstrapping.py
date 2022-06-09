import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib


#%% Figure 1: Boundaries of the NEON Domains with Stations and Histogram
# loading NEON domain shapefile
neon_gdf = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NEONDomains_0/NEON_Domains.shp')

# loading and filtering station data
ghcn_stations = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/NEON_domain_daily_precip_stats.csv')
qual_stations = ghcn_stations[~np.isnan(ghcn_stations['early_late'])]

num_domains = 17
domains = np.arange(1, num_domains+1)
early_count = np.zeros([num_domains])
late_count = np.zeros([num_domains])
both_count = np.zeros([num_domains])

for domain in domains:
    early_count[domain-1] = np.sum((qual_stations['early_late']==1) & (qual_stations['station_domain']==domain))
    late_count[domain-1] = np.sum((qual_stations['early_late']==2) & (qual_stations['station_domain']==domain))
    both_count[domain-1] = np.sum((qual_stations['early_late']==3) & (qual_stations['station_domain']==domain))

# prepping plotting boundaries
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
crs_new = ccrs.PlateCarree()

# plotting country, state boundaries
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(17, 11))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray', linewidth=0.5)
ax.set_aspect('equal')
# plotting NEON domain boundaries
neon_gdf.plot(ax=ax, color='white', edgecolor='black')  # plotting neon boundaries
plt.scatter(
    x = qual_stations['station_lon'][qual_stations['early_late']==3],
    y = qual_stations['station_lat'][qual_stations['early_late']==3],
    color='royalblue',
    marker = '.',
    s = 10,
    alpha = 0.9,
    transform=ccrs.PlateCarree()
)
# framing around US
plt.xlim([-129, -62])
plt.ylim([23, 52])
plt.xticks(np.arange(-125, -55, 10), fontsize=18)
plt.yticks(np.arange(30, 60, 10), fontsize=18)
plt.title('NEON Ecoregions and GHCN Station Locations', fontsize=24)
plt.show()


stations_with_domain = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/NEON_domain_daily_precip_stats.csv')
domain_station_counts = stations_with_domain[stations_with_domain['early_late']==3].groupby('station_domain').count()
domain_station_counts = domain_station_counts['early_late']

fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(np.arange(1, 18, 1), domain_station_counts[0:17],
        facecolor=matplotlib.colors.to_rgba('royalblue', 0.67), edgecolor='black')
ax.invert_yaxis()
# ax.invert_xaxis()
plt.yticks(np.arange(1, 18, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(0, 350, 50), fontsize=20)
ax.axvline(color='k')
plt.xlabel('count', fontsize=20)
plt.title('Stations per Region', fontsize=24)
plt.show()




#%% Figure 2: Map and Bar Chart of Precipitation Distribution Shift
# figure 1a
NEON_domains = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NEONDomains_0/NEON_Domains.shp')
NEON_domains = NEON_domains.drop([1, 5], axis = 0)  # dropping 'extra' domain portions for centroid graphing purposes
NEON_domains.index = np.arange(1, 21, 1)
NEON_domains = NEON_domains.drop([18, 19, 20], axis = 0)

pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NEON_v2/block_bootstrap_results/results/domain_bootstrapped_daily_precipitation_pdf_moments_merged.csv')
pdf_results = pdf_results.drop([17, 18, 19], axis = 0)

pdf_mean_change = pdf_results['mean_diff_mid']
pdf_mean_change.index = pdf_results['regions']
pdf_mean_change_sig = pdf_results['mean_diff_low'] > 0  # can use this as a significance test since none of the results are significant on the low end
pdf_mean_change_sig.index = pdf_results['regions']

crs_new = ccrs.PlateCarree()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')


# fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(14, 8))
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(17, 11))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='lightgray')
c_val = np.max([abs(np.min(pdf_mean_change)), abs(np.max(pdf_mean_change))])
cmap = 'bwr_r'
NEON_domains.plot(column = pdf_mean_change, ax = ax, cmap=cmap, edgecolor='k', vmin=-c_val, vmax=c_val, linewidth=2)
pdf_mean_change[~pdf_mean_change_sig] = np.nan
NEON_domains.plot(column = pdf_mean_change, ax = ax, cmap=cmap, edgecolor='k', vmin=-c_val, vmax=c_val, missing_kwds={'color':'none', 'edgecolor':'grey', 'hatch':'/'})
pdf_mean_change[:] = np.nan
NEON_domains.plot(column = pdf_mean_change, ax = ax, cmap=cmap, edgecolor='k', vmin=-c_val, vmax=c_val, missing_kwds={'color':'none', 'edgecolor':'k'})
plt.set_cmap(cmap)
sm = plt.cm.ScalarMappable(cmap='bwr_r', norm=plt.Normalize(-c_val, c_val))
cbar = fig.colorbar(sm, fraction=0.0275, pad=0.04)
cbar.set_label('change in mean wet day precipitation (mm)', fontsize=18)
cbar.ax.tick_params(labelsize=14)
plt.xlim([-129, -62])
plt.ylim([23, 52])
plt.xticks(np.arange(-125, -55, 10), fontsize=18)
plt.yticks(np.arange(30, 60, 10), fontsize=18)
plt.title('Change in Mean Wet Day Intensity', fontsize=24)
plt.show()

# figures 1b-c
pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NEON_v2/block_bootstrap_results/results/domain_bootstrapped_daily_precipitation_pdf_moments_merged.csv')
pdf_results = pdf_results.drop([17, 18, 19], axis = 0)
pdf_results_norm = pdf_results.copy()

pdf_results_norm['mean_low'] = pdf_results_norm['mean_diff_low']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['mean_mid'] = pdf_results_norm['mean_diff_mid']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['mean_high'] = pdf_results_norm['mean_diff_high']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['median_low'] = pdf_results_norm['median_diff_low']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['median_mid'] = pdf_results_norm['median_diff_mid']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['median_high'] = pdf_results_norm['median_diff_high']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['std_diff_low'] = pdf_results_norm['std_diff_low']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['std_diff_mid'] = pdf_results_norm['std_diff_mid']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['std_diff_high'] = pdf_results_norm['std_diff_high']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['skew_diff_low'] = pdf_results_norm['skew_diff_low']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['skew_diff_mid'] = pdf_results_norm['skew_diff_mid']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['skew_diff_high'] = pdf_results_norm['skew_diff_high']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['kurt_diff_low'] = pdf_results_norm['kurt_diff_low']/pdf_results_norm['pdf_1_kurt']*100
pdf_results_norm['kurt_diff_mid'] = pdf_results_norm['kurt_diff_mid']/pdf_results_norm['pdf_1_kurt']*100
pdf_results_norm['kurt_diff_high'] = pdf_results_norm['kurt_diff_high']/pdf_results_norm['pdf_1_kurt']*100


fig, ax = plt.subplots(figsize=(8, 12))
asymmetric_err = [pdf_results_norm['mean_mid'] - pdf_results_norm['mean_low'], pdf_results_norm['mean_high'] - pdf_results_norm['mean_mid']]
ax.barh(np.arange(1, 18, 1), pdf_results_norm['mean_mid'], xerr=np.array(asymmetric_err),
        facecolor=matplotlib.colors.to_rgba('royalblue', 0.67), edgecolor='black')
ax.invert_yaxis()
plt.yticks(np.arange(1, 18, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(-5, 15, 5), fontsize=20)
ax.axvline(color='k')
plt.xlabel('change in mean (%)', fontsize=20)
plt.title('Change in Mean', fontsize=24)
plt.show()


fig, ax = plt.subplots(figsize=(8, 12))
asymmetric_err = [pdf_results_norm['std_diff_mid'] - pdf_results_norm['std_diff_low'], pdf_results_norm['std_diff_high'] - pdf_results_norm['std_diff_mid']]
ax.barh(np.arange(1, 18, 1), pdf_results_norm['std_diff_mid'], xerr=np.array(asymmetric_err),
        facecolor=matplotlib.colors.to_rgba('firebrick', 0.67), edgecolor='black')
ax.invert_yaxis()
plt.yticks(np.arange(1, 18, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(-10, 25, 5), fontsize=20)
plt.xlim([-14, 14])
ax.axvline(color='k')
plt.xlabel('change in standard deviation (%)', fontsize=20)
plt.title('Change in Standard Deviation', fontsize=24)
plt.show()





#%% Figure 3: Creating stylized versions of each domain's precip trends
for domain in np.arange(1, 18, 1):  # looping over each domain
    # domain = 1
    pdf = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NEON_v2/block_bootstrap_results/pdf_shift/pdf_shift_percentiles_domain_' + str(domain) + '.npy')

    num_bins = np.shape(pdf)[1]+1
    pdf_low = np.zeros(num_bins + 3); pdf_low[:] = np.nan
    pdf_low[2:-2] = pdf[0, :]
    pdf_mid = np.zeros(num_bins + 3); pdf_mid[:] = np.nan
    pdf_mid[2:-2] = pdf[1, :]
    pdf_high = np.zeros(num_bins + 3); pdf_high[:] = np.nan
    pdf_high[2:-2] = pdf[2, :]

    pdf_low_rolling = pd.Series(pdf_low).rolling(3, min_periods=1).mean()[4:]
    pdf_mid_rolling = pd.Series(pdf_mid).rolling(3, min_periods=1).mean()[4:]
    pdf_high_rolling = pd.Series(pdf_high).rolling(3, min_periods=1).mean()[4:]
    smoothing_x = np.linspace(0, 100, num_bins)
    pfit_low = np.polyfit(smoothing_x[:-1], pdf_low_rolling, 5)
    pfit_mid = np.polyfit(smoothing_x[:-1], pdf_mid_rolling, 5)
    pfit_high = np.polyfit(smoothing_x[:-1], pdf_high_rolling, 5)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.linspace(0, 100, 500)
    plt.plot(x, np.polyval(pfit_low, x), 'k', linewidth=1)
    plt.plot(x, np.polyval(pfit_mid, x), 'k', linewidth=3)
    plt.plot(x, np.polyval(pfit_high, x), 'k', linewidth=1)
    ax.fill_between(x, np.polyval(pfit_low, x), np.polyval(pfit_high, x), alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([0, 100], ['0', '100'], fontsize=24)
    plt.ylim([-20, 30])
    plt.yticks([-15, 0, 15, 30], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.arange(0, 100, 5)
    plt.plot(x, pdf_low[2:-2], 'k', linewidth=1)
    plt.plot(x, pdf_mid[2:-2], 'k', linewidth=3)
    plt.plot(x, pdf_high[2:-2], 'k', linewidth=1)
    ax.fill_between(x, pdf_low[2:-2], pdf_high[2:-2], alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([0, 100], ['0', '100'], fontsize=24)
    plt.ylim([-20, 30])
    plt.yticks([-15, 0, 15, 30], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()


    # for the over-99 crowd
    pdf = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NEON_v2/block_bootstrap_results/pdf_shift/pdf_shift_percentiles_99_domain_' + str(domain) + '.npy')

    num_bins = np.shape(pdf)[1]+1
    pdf_low = np.zeros(num_bins + 3); pdf_low[:] = np.nan
    pdf_low[2:-2] = pdf[0, :]
    pdf_mid = np.zeros(num_bins + 3); pdf_mid[:] = np.nan
    pdf_mid[2:-2] = pdf[1, :]
    pdf_high = np.zeros(num_bins + 3); pdf_high[:] = np.nan
    pdf_high[2:-2] = pdf[2, :]

    # come back to this minimum periods thing
    pdf_low_rolling = pd.Series(pdf_low).rolling(3, min_periods=1).mean()[4:]
    pdf_mid_rolling = pd.Series(pdf_mid).rolling(3, min_periods=1).mean()[4:]
    pdf_high_rolling = pd.Series(pdf_high).rolling(3, min_periods=1).mean()[4:]
    smoothing_x = np.linspace(99, 100, num_bins)
    pfit_low = np.polyfit(smoothing_x[:-1], pdf_low_rolling, 5)
    pfit_mid = np.polyfit(smoothing_x[:-1], pdf_mid_rolling, 5)
    pfit_high = np.polyfit(smoothing_x[:-1], pdf_high_rolling, 5)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.linspace(99, 100, 500)
    plt.plot(x, np.polyval(pfit_low, x), 'k', linewidth=1)
    plt.plot(x, np.polyval(pfit_mid, x), 'k', linewidth=3)
    plt.plot(x, np.polyval(pfit_high, x), 'k', linewidth=1)
    ax.fill_between(x, np.polyval(pfit_low, x), np.polyval(pfit_high, x), alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([99, 100], ['99', '100'], fontsize=24)
    plt.ylim([-75, 75])
    plt.yticks([-75, 0, 75], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.arange(99, 100, 0.05)
    plt.plot(x, pdf_low[2:-2], 'k', linewidth=1)
    plt.plot(x, pdf_mid[2:-2], 'k', linewidth=3)
    plt.plot(x, pdf_high[2:-2], 'k', linewidth=1)
    ax.fill_between(x, pdf_low[2:-2], pdf_high[2:-2], alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([99, 100], ['99', '100'], fontsize=24)
    plt.ylim([-75, 75])
    plt.yticks([-75, 0, 75], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()



# plotting domains for background
# loading NEON domain shapefile
neon_gdf = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NEONDomains_0/NEON_Domains.shp')
neon_gdf['shading_east'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
neon_gdf['shading_west'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
crs_new = ccrs.PlateCarree()

# Full US
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(17, 11))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray', linewidth=0.5)
ax.set_aspect('equal')
# plotting NEON domain boundaries
neon_gdf.plot(column='shading_east', ax=ax, edgecolor='black', cmap='Blues', alpha=0.5)  # plotting neon boundaries
neon_gdf.plot(column='shading_west', ax=ax, edgecolor='black', cmap='Greys', alpha=0.1)  # plotting neon boundaries
plt.xlim([-129, -62])
plt.ylim([23, 52])
plt.xticks(np.arange(-125, -55, 10), fontsize=18)
plt.yticks(np.arange(30, 60, 10), fontsize=18)
plt.show()




#%% Supporting Information Figure 1: Boundaries of the NCA Regions with Stations and Histogram
# loading NEON domain shapefile
nca_gdf = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NCARegions/cb_2020_us_state_500k_ncaregions.shp')

# loading and filtering station data
ghcn_stations = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/NCA_region_daily_precip_stats.csv')
qual_stations = ghcn_stations[~np.isnan(ghcn_stations['early_late'])]

num_regions = 10
regions = np.arange(1, num_regions+1)
early_count = np.zeros([num_regions])
late_count = np.zeros([num_regions])
both_count = np.zeros([num_regions])

for region in regions:
    early_count[region-1] = np.sum((qual_stations['early_late']==1) & (qual_stations['station_region']==region))
    late_count[region-1] = np.sum((qual_stations['early_late']==2) & (qual_stations['station_region']==region))
    both_count[region-1] = np.sum((qual_stations['early_late']==3) & (qual_stations['station_region']==region))

# prepping plotting boundaries
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
crs_new = ccrs.PlateCarree()

# plotting country, state boundaries
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(17, 11))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray', linewidth=0.5)
ax.set_aspect('equal')
# plotting NCA domain boundaries
nca_gdf.plot(ax=ax, color='white', edgecolor='black')  # plotting neon boundaries
plt.scatter(
    x = qual_stations['station_lon'][qual_stations['early_late']==3],
    y = qual_stations['station_lat'][qual_stations['early_late']==3],
    color='royalblue',
    marker = '.',
    s = 10,
    alpha = 0.9,
    transform=ccrs.PlateCarree()
)
# framing around US
plt.xlim([-129, -62])
plt.ylim([23, 52])
plt.xticks(np.arange(-125, -55, 10), fontsize=18)
plt.yticks(np.arange(30, 60, 10), fontsize=18)
plt.title('NCA Regions and GHCN Station Locations', fontsize=24)
plt.show()


stations_with_domain = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/NCA_region_daily_precip_stats.csv')
domain_station_counts = stations_with_domain[stations_with_domain['early_late']==3].groupby('station_region').count()
domain_station_counts = domain_station_counts['early_late']

fig, ax = plt.subplots(figsize=(8, 10))
ax.barh(np.arange(4, 11, 1), domain_station_counts[4:11], height = 0.7,
        facecolor=matplotlib.colors.to_rgba('royalblue', 0.67), edgecolor='black')
ax.invert_yaxis()
plt.yticks(np.arange(4, 11, 1))
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(0, 501, 100), fontsize=20)
ax.axvline(color='k')
plt.xlabel('count', fontsize=20)
plt.title('Stations per Region', fontsize=24)
plt.show()





#%% Supprorting Info Figure S3: Graphically looking at changes in intensity
import fnmatch
import os
import os.path

def bootstrap_pdf_creator(domain_num):

    # looping over first half years
    domain_first_half_pdf = np.array([])

    for yr in np.arange(1951, 1981, 2):
        fpath = '/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/bootstrapping_blocks/domain_bootstrapping/domain_' + str(domain_num) + '/first_half/' + str(yr) + '_' + str(yr+1) + '/'
        # fpath = '/home/rdh0715/domain_bootstrapping/domain_' + str(domain_num) + '/first_half/' + str(yr) + '_' + str(yr+1) + '/'
        block_file_list = fnmatch.filter(os.listdir(fpath), '*.npy')
        all_blocks = np.array([np.load(fpath + fname) for fname in block_file_list])
        num_blocks = len(block_file_list)  #! Might need to come back to this > should it be the smallest number of block segments?
        rand_block_nums = np.random.choice(np.arange(num_blocks), size = num_blocks, replace = True)
        for block_num in rand_block_nums:
            domain_first_half_pdf = np.concatenate((domain_first_half_pdf, all_blocks[block_num]))


    # looping over second half years
    domain_second_half_pdf = np.array([])

    for yr in np.arange(1991, 2021, 2):
        fpath = '/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/bootstrapping_blocks/domain_bootstrapping/domain_' + str(domain_num) + '/second_half/' + str(yr) + '_' + str(yr+1) + '/'
        # fpath = '/home/rdh0715/domain_bootstrapping/domain_' + str(domain_num) + '/second_half/' + str(yr) + '_' + str(yr+1) + '/'
        block_file_list = fnmatch.filter(os.listdir(fpath), '*.npy')
        all_blocks = np.array([np.load(fpath + fname) for fname in block_file_list])
        num_blocks = len(block_file_list)  #! Might need to come back to this > should it be the smallest number of block segments?
        rand_block_nums = np.random.choice(np.arange(num_blocks), size = num_blocks, replace = True)
        for block_num in rand_block_nums:
            domain_second_half_pdf = np.concatenate((domain_second_half_pdf, all_blocks[block_num]))


    # clearing out nans for future calculations
    domain_first_half_pdf = domain_first_half_pdf[~np.isnan(domain_first_half_pdf)]
    domain_second_half_pdf = domain_second_half_pdf[~np.isnan(domain_second_half_pdf)]

    return domain_first_half_pdf, domain_second_half_pdf


domain = 5  # using Great Lakes as an example
pdf_1_both, pdf_2_both = bootstrap_pdf_creator(domain)
domain_names = ['Northeast', 'Mid Atlantic', 'Southeast', 'Atlantic Neotropical', 'Great Lakes', 'Prairie Peninsula', 'Appalachians and Cumberland Plateau', 'Ozarks Complex', 'Northern Plains', 'Central Plains', 'Southern Plains', 'Northern Rockies', 'Southern Rockies and Colorado Plateau', 'Desert Southwest', 'Great Basin', 'Pacific Northwest', 'Pacific Southwest']
domain_name = domain_names[domain-1]

percentile_bins = np.array([])
for b in np.arange(0, 1.01, 0.05):
    percentile_bins = np.append(percentile_bins, np.round(np.quantile(pdf_1_both, q=b), 1))
percentile_bins[-1] = np.inf
percentile_bins = np.sort(percentile_bins)
pdf_1_hist_density_percentiles = np.histogram(pdf_1_both, bins=percentile_bins)
pdf_2_hist_density_percentiles = np.histogram(pdf_2_both, bins=percentile_bins)
pdf_1_hist_density_percentiles_norm = pdf_1_hist_density_percentiles[0] / np.nansum(pdf_1_hist_density_percentiles[0])
pdf_2_hist_density_percentiles_norm = pdf_2_hist_density_percentiles[0] / np.nansum(pdf_2_hist_density_percentiles[0])
pdf_shift_percentiles = (pdf_2_hist_density_percentiles_norm - pdf_1_hist_density_percentiles_norm)
pdf_shift_percentiles_norm = ((pdf_2_hist_density_percentiles_norm - pdf_1_hist_density_percentiles_norm) / pdf_1_hist_density_percentiles_norm) * 100


# side-by-side histograms of early and late periods
num_bins = 25
per_99 = np.quantile(pdf_1_both, q=0.99)
bin_size_low = np.linspace(1, per_99, num_bins)[1] - np.linspace(1, per_99, num_bins)[0]
pdf_1_hist_density = np.histogram(pdf_1_both, bins=np.linspace(1, per_99, num_bins), density='True')
fig, ax = plt.subplots(figsize=(14, 8))
pdf_2_hist_density = np.histogram(pdf_2_both, bins=np.linspace(1, per_99, num_bins), density='True')
plt.bar(pdf_1_hist_density[1][:-1]-0.3*bin_size_low/2, pdf_1_hist_density[0]*100, facecolor=matplotlib.colors.to_rgba('forestgreen', 0.25), edgecolor='black', width=0.3*bin_size_low)
plt.bar(pdf_1_hist_density[1][:-1]+0.3*bin_size_low/2, pdf_2_hist_density[0]*100, facecolor=matplotlib.colors.to_rgba('royalblue', 0.9), edgecolor='black', width=0.3*bin_size_low)
plt.xticks(fontsize=22)
plt.yticks(np.arange(0, 40, 5), fontsize=22)
# plt.ylim([0, np.ceil(np.max(pdf_1_hist_density[0])*100+1)])
plt.ylim([0, 20])
plt.ylabel('frequencies (%)', fontsize=24)
plt.xlabel('mm/wet day', fontsize=24)
# plt.title('Great Lakes', fontsize=28)
plt.show()


# plot difference in distributions at each bin
num_bins = np.shape(pdf)[1] + 1
fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(np.arange(0, 100, 5), pdf_shift_percentiles * 100, align='edge', facecolor=matplotlib.colors.to_rgba('royalblue', 0.75), edgecolor='black', width=4)
plt.axhline(y=0, color='k')
plt.xticks(fontsize=22)
plt.yticks(np.arange(-0.6, 1, 0.3), fontsize=22)
# plt.ylim([0, np.ceil(np.max(pdf_1_hist_density[0])*100+1)])
plt.ylabel('frequency change (%)', fontsize=24)
plt.xlabel('percentile', fontsize=24)
# plt.title('domain ' + str(domain) + ': sub-99th percentile')
# plt.title('Intensity Changes', fontsize=28)
plt.show()


# plot normalized difference in distributions at each bin for sub-99th percentiles
fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(np.arange(0, 100, 5), pdf_shift_percentiles_norm, align='edge', facecolor=matplotlib.colors.to_rgba('royalblue', 0.75), edgecolor='black', width=4)
pdf_rolling = pd.Series(pdf_shift_percentiles_norm).rolling(3, min_periods=1, center=True).mean()[1:]
smoothing_x = np.linspace(0, 100, num_bins-1)
pfit = np.polyfit(smoothing_x[:-1], pdf_rolling, 5)
x = np.linspace(0, 100, 500)
plt.plot(x, np.polyval(pfit, x), 'k', linewidth=3)
plt.axhline(y=0, color='k')
plt.xticks(fontsize=22)
plt.ylim([-12, 20])
plt.yticks(np.arange(-10, 21, 10), fontsize=22)
plt.ylabel('relative frequency change (%)', fontsize=24)
plt.xlabel('percentile', fontsize=24)
# plt.title('Relative Intensity Changes', fontsize=28)
plt.show()





#%% Supplemental Figure 2: Map and Bar Chart of Precipitation Distribution Shift for NCA Regions
# figure S2a
NCA_regions = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NCARegions/cb_2020_us_state_500k_ncaregions.shp')
NCA_regions = NCA_regions.drop([0, 1, 2], axis = 0)  # dropping 'extra' domain portions for centroid graphing purposes
NCA_regions.index = np.arange(4, 11, 1)

pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NCA_v2/block_bootstrap_results_NCA/results/bootstrapped_daily_precipitation_pdf_moments_merged.csv')

pdf_mean_change = pdf_results['mean_diff_mid'][3:]
pdf_mean_change.index = pdf_results['regions'][3:]
pdf_mean_change_sig = pdf_results['mean_diff_low'] > 0
pdf_mean_change_sig.index = pdf_results['regions']

crs_new = ccrs.PlateCarree()
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')


fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(17, 11))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='lightgray')
c_val = np.max([abs(np.min(pdf_mean_change)), abs(np.max(pdf_mean_change))])
cmap = 'bwr_r'
NCA_regions.plot(column = pdf_mean_change, ax = ax, cmap=cmap, edgecolor='k', vmin=-c_val, vmax=c_val, linewidth=2)
pdf_mean_change[~pdf_mean_change_sig] = np.nan
NCA_regions.plot(column = pdf_mean_change, ax = ax, cmap=cmap, edgecolor='k', vmin=-c_val, vmax=c_val, missing_kwds={'color':'none', 'edgecolor':'grey', 'hatch':'/'})
pdf_mean_change[:] = np.nan
NCA_regions.plot(column = pdf_mean_change, ax = ax, cmap=cmap, edgecolor='k', vmin=-c_val, vmax=c_val, missing_kwds={'color':'none', 'edgecolor':'k'})
plt.set_cmap(cmap)
sm = plt.cm.ScalarMappable(cmap='bwr_r', norm=plt.Normalize(-c_val, c_val))
cbar = fig.colorbar(sm, fraction=0.025, pad=0.04)
cbar.set_label('change in mean wet day precipitation (mm)', fontsize=18)
cbar.ax.tick_params(labelsize=14)
plt.xlim([-129, -62])
plt.ylim([23, 52])
plt.xticks(np.arange(-125, -55, 10), fontsize=18)
plt.yticks(np.arange(30, 60, 10), fontsize=18)
plt.title('Change in Mean Wet Day Intensity', fontsize=24)
plt.show()

# figures 1b-c
pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NCA_v2/block_bootstrap_results_NCA/results/bootstrapped_daily_precipitation_pdf_moments_merged.csv')
pdf_results_norm = pdf_results[3:].copy()
pdf_results_norm['mean_low'] = pdf_results_norm['mean_diff_low']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['mean_mid'] = pdf_results_norm['mean_diff_mid']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['mean_high'] = pdf_results_norm['mean_diff_high']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['median_low'] = pdf_results_norm['median_diff_low']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['median_mid'] = pdf_results_norm['median_diff_mid']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['median_high'] = pdf_results_norm['median_diff_high']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['std_diff_low'] = pdf_results_norm['std_diff_low']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['std_diff_mid'] = pdf_results_norm['std_diff_mid']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['std_diff_high'] = pdf_results_norm['std_diff_high']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['skew_diff_low'] = pdf_results_norm['skew_diff_low']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['skew_diff_mid'] = pdf_results_norm['skew_diff_mid']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['skew_diff_high'] = pdf_results_norm['skew_diff_high']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['kurt_diff_low'] = pdf_results_norm['kurt_diff_low']/pdf_results_norm['pdf_1_kurt']*100
pdf_results_norm['kurt_diff_mid'] = pdf_results_norm['kurt_diff_mid']/pdf_results_norm['pdf_1_kurt']*100
pdf_results_norm['kurt_diff_high'] = pdf_results_norm['kurt_diff_high']/pdf_results_norm['pdf_1_kurt']*100


fig, ax = plt.subplots(figsize=(10, 10))
asymmetric_err = [pdf_results_norm['mean_mid'] - pdf_results_norm['mean_low'], pdf_results_norm['mean_high'] - pdf_results_norm['mean_mid']]
ax.barh(np.arange(1, 8, 1), pdf_results_norm['mean_mid'], xerr=np.array(asymmetric_err),
        facecolor=matplotlib.colors.to_rgba('royalblue', 0.67), edgecolor='black')
ax.invert_yaxis()
plt.yticks(np.arange(1, 8, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(-5, 15, 5), fontsize=20)
ax.axvline(color='k')
plt.xlabel('change in mean (%)', fontsize=20)
plt.title('Change in Mean', fontsize=28)
plt.show()


fig, ax = plt.subplots(figsize=(10, 10))
asymmetric_err = [pdf_results_norm['std_diff_mid'] - pdf_results_norm['std_diff_low'], pdf_results_norm['std_diff_high'] - pdf_results_norm['std_diff_mid']]
ax.barh(np.arange(1, 8, 1), pdf_results_norm['std_diff_mid'], xerr=np.array(asymmetric_err),
        facecolor=matplotlib.colors.to_rgba('firebrick', 0.67), edgecolor='black')
ax.invert_yaxis()
plt.yticks(np.arange(1, 8, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(-10, 12, 5), fontsize=20)
plt.xlim([-12, 12])
ax.axvline(color='k')
plt.xlabel('change in standard deviation (%)', fontsize=20)
plt.title('Change in Standard Deviation', fontsize=28)
plt.show()




#%% Supplemental Figure X: Creating stylized versions of each region's precip trends
for domain in np.arange(4, 11, 1):  # looping over each region
    domain = 5
    pdf = np.load(
        '/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NCA_v2/block_bootstrap_results_NCA/pdf_shift/pdf_shift_percentiles_region_' + str(
            domain) + '.npy')

    num_bins = np.shape(pdf)[1] + 1
    pdf_low = np.zeros(num_bins + 3);
    pdf_low[:] = np.nan
    pdf_low[2:-2] = pdf[0, :]
    pdf_mid = np.zeros(num_bins + 3);
    pdf_mid[:] = np.nan
    pdf_mid[2:-2] = pdf[1, :]
    pdf_high = np.zeros(num_bins + 3);
    pdf_high[:] = np.nan
    pdf_high[2:-2] = pdf[2, :]

    # come back to this minimum periods thing
    pdf_low_rolling = pd.Series(pdf_low).rolling(3, min_periods=1, center=True).mean()[2:-2]
    pdf_mid_rolling = pd.Series(pdf_mid).rolling(3, min_periods=1, center=True).mean()[2:-2]
    pdf_high_rolling = pd.Series(pdf_high).rolling(3, min_periods=1, center=True).mean()[2:-2]
    smoothing_x = np.linspace(0, 100, num_bins)
    pfit_low = np.polyfit(smoothing_x[:-1], pdf_low_rolling, 5)
    pfit_mid = np.polyfit(smoothing_x[:-1], pdf_mid_rolling, 5)
    pfit_high = np.polyfit(smoothing_x[:-1], pdf_high_rolling, 5)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.linspace(0, 100, 500)
    plt.plot(x, np.polyval(pfit_low, x), 'k', linewidth=1)
    plt.plot(x, np.polyval(pfit_mid, x), 'k', linewidth=3)
    plt.plot(x, np.polyval(pfit_high, x), 'k', linewidth=1)
    ax.fill_between(x, np.polyval(pfit_low, x), np.polyval(pfit_high, x), alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([0, 100], ['0', '100'], fontsize=24)
    plt.ylim([-20, 20])
    plt.yticks([-20, 0, 20], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.arange(0, 100, 5)
    plt.plot(x, pdf_low[2:-2], 'k', linewidth=1)
    plt.plot(x, pdf_mid[2:-2], 'k', linewidth=3)
    plt.plot(x, pdf_high[2:-2], 'k', linewidth=1)
    ax.fill_between(x, pdf_low[2:-2], pdf_high[2:-2], alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([0, 100], ['0', '100'], fontsize=24)
    plt.ylim([-20, 20])
    plt.yticks([-20, 0, 20], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()


    # for the over-99 crowd
    pdf = np.load(
        '/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NCA_v2/block_bootstrap_results_NCA/pdf_shift/pdf_shift_percentiles_99_region_' + str(
            domain) + '.npy')

    num_bins = np.shape(pdf)[1] + 1
    pdf_low = np.zeros(num_bins + 3);
    pdf_low[:] = np.nan
    pdf_low[2:-2] = pdf[0, :]
    pdf_mid = np.zeros(num_bins + 3);
    pdf_mid[:] = np.nan
    pdf_mid[2:-2] = pdf[1, :]
    pdf_high = np.zeros(num_bins + 3);
    pdf_high[:] = np.nan
    pdf_high[2:-2] = pdf[2, :]

    # come back to this minimum periods thing
    pdf_low_rolling = pd.Series(pdf_low).rolling(3, min_periods=1).mean()[4:]
    pdf_mid_rolling = pd.Series(pdf_mid).rolling(3, min_periods=1).mean()[4:]
    pdf_high_rolling = pd.Series(pdf_high).rolling(3, min_periods=1).mean()[4:]
    smoothing_x = np.linspace(99, 100, num_bins)
    pfit_low = np.polyfit(smoothing_x[:-1], pdf_low_rolling, 5)
    pfit_mid = np.polyfit(smoothing_x[:-1], pdf_mid_rolling, 5)
    pfit_high = np.polyfit(smoothing_x[:-1], pdf_high_rolling, 5)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.linspace(99, 100, 500)
    plt.plot(x, np.polyval(pfit_low, x), 'k', linewidth=1)
    plt.plot(x, np.polyval(pfit_mid, x), 'k', linewidth=3)
    plt.plot(x, np.polyval(pfit_high, x), 'k', linewidth=1)
    ax.fill_between(x, np.polyval(pfit_low, x), np.polyval(pfit_high, x), alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([99, 100], ['99', '100'], fontsize=24)
    plt.ylim([-75, 75])
    plt.yticks([-75, 0, 75], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.arange(99, 100, 0.05)
    plt.plot(x, pdf_low[2:-2], 'k', linewidth=1)
    plt.plot(x, pdf_mid[2:-2], 'k', linewidth=3)
    plt.plot(x, pdf_high[2:-2], 'k', linewidth=1)
    ax.fill_between(x, pdf_low[2:-2], pdf_high[2:-2], alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([99, 100], ['99', '100'], fontsize=24)
    plt.ylim([-75, 75])
    plt.yticks([-75, 0, 75], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()




    pdf = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/results/block_bootstrap_results_NCA/pdf_shift/pdf_shift_region_' + str(domain) + '.npy')
    pdf_low = np.zeros(28); pdf_low[:] = np.nan
    pdf_low[2:-2] = pdf[0, :]
    pdf_mid = np.zeros(28); pdf_mid[:] = np.nan
    pdf_mid[2:-2] = pdf[1, :]
    pdf_high = np.zeros(28); pdf_high[:] = np.nan
    pdf_high[2:-2] = pdf[2, :]

    num_bins = 25
    pdf_low_rolling = pd.Series(pdf_low).rolling(5, min_periods=3).mean()[4:]
    pdf_mid_rolling = pd.Series(pdf_mid).rolling(5, min_periods=3).mean()[4:]
    pdf_high_rolling = pd.Series(pdf_high).rolling(5, min_periods=3).mean()[4:]
    smoothing_x = np.linspace(0, 99, num_bins)
    pfit_low = np.polyfit(smoothing_x[:-1], pdf_low_rolling, 4)
    pfit_mid = np.polyfit(smoothing_x[:-1], pdf_mid_rolling, 4)
    pfit_high = np.polyfit(smoothing_x[:-1], pdf_high_rolling, 4)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    x = np.linspace(0, 99, 500)
    plt.plot(x, np.polyval(pfit_low, x), 'k', linewidth=1)
    plt.plot(x, np.polyval(pfit_mid, x), 'k', linewidth=3)
    plt.plot(x, np.polyval(pfit_high, x), 'k', linewidth=1)
    ax.fill_between(x, np.polyval(pfit_low, x), np.polyval(pfit_high, x), alpha=0.2)
    plt.axhline(y=0, color='k')
    plt.xticks([0, 99], ['0th', '99th'], fontsize=24)
    plt.ylim([-30, 65])
    plt.yticks([-25, 0, 25, 50], fontsize=24)
    # plt.title('domain ' + str(domain))
    plt.show()


# plotting domains for background
# loading NEON domain shapefile
NCA_regions = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NCARegions/cb_2020_us_state_500k_ncaregions.shp')
NCA_regions = NCA_regions.drop([0, 1, 2], axis = 0)  # dropping 'extra' domain portions for centroid graphing purposes
NCA_regions.index = np.arange(4, 11, 1)
NCA_regions['shading_east'] = [1, 1, 1, 0, 1, 1, 0]
NCA_regions['shading_west'] = [0, 0, 0, 1, 0, 0, 1]
states_provinces = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces_lines',
    scale='50m',
    facecolor='none')
crs_new = ccrs.PlateCarree()

# Full US
fig, ax = plt.subplots(1, 1, subplot_kw={'projection': crs_new}, figsize=(17, 11))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(states_provinces, edgecolor='gray', linewidth=0.5)
ax.set_aspect('equal')
# plotting NEON domain boundaries
NCA_regions.plot(column='shading_east', ax=ax, edgecolor='black', cmap='Blues', alpha=0.5)
NCA_regions.plot(column='shading_west', ax=ax, edgecolor='black', cmap='Greys', alpha=0.1)  # plotting neon boundaries
plt.xlim([-129, -62])
plt.ylim([23, 52])
plt.xticks(np.arange(-125, -55, 10), fontsize=18)
plt.yticks(np.arange(30, 60, 10), fontsize=18)
plt.show()
