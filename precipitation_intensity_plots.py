import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib


#%% Figure 1: Map and Bar Chart of Precipitation Distribution Shift
# figure 1a
NEON_domains = gpd.read_file('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/NEONDomains_0/NEON_Domains.shp')
NEON_domains = NEON_domains.drop([1, 5], axis = 0)  # dropping 'extra' domain portions for centroid graphing purposes
NEON_domains.index = np.arange(1, 21, 1)
NEON_domains = NEON_domains.drop([18, 19, 20], axis = 0)

pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_daily_pdf_moments_both_median.csv')
# pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_daily_pdf_moments_first_and_second.csv')

pdf_mean_change = pdf_results['mean_mid']
pdf_mean_change.index = pdf_results['regions']

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
# pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_pdf_moments.csv')
pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_daily_pdf_moments_both_median.csv')
# pdf_results = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_daily_pdf_moments_first_and_second.csv')
pdf_results_norm = pdf_results.copy()
pdf_results_norm['mean_low'] = pdf_results_norm['mean_low']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['mean_mid'] = pdf_results_norm['mean_mid']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['mean_high'] = pdf_results_norm['mean_high']/pdf_results_norm['pdf_1_mean']*100
pdf_results_norm['median_low'] = pdf_results_norm['median_low']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['median_mid'] = pdf_results_norm['median_mid']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['median_high'] = pdf_results_norm['median_high']/pdf_results_norm['pdf_1_median']*100
pdf_results_norm['std_diff_low'] = pdf_results_norm['std_diff_low']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['std_diff_mid'] = pdf_results_norm['std_diff_mid']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['std_diff_high'] = pdf_results_norm['std_diff_high']/pdf_results_norm['pdf_1_std']*100
pdf_results_norm['skew_diff_low'] = pdf_results_norm['skew_diff_low']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['skew_diff_mid'] = pdf_results_norm['skew_diff_mid']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['skew_diff_high'] = pdf_results_norm['skew_diff_high']/pdf_results_norm['pdf_1_skew']*100
pdf_results_norm['kurt_diff_low'] = pdf_results_norm['kurt_diff_low']/pdf_results_norm['pdf_1_kurt']*100
pdf_results_norm['kurt_diff_mid'] = pdf_results_norm['kurt_diff_mid']/pdf_results_norm['pdf_1_kurt']*100
pdf_results_norm['kurt_diff_high'] = pdf_results_norm['kurt_diff_high']/pdf_results_norm['pdf_1_kurt']*100

# domain_name = ['Northeast', 'Mid Atlantic', 'Southeast', 'Atlantic Neotropical', 'Great Lakes', 'Prairie Peninsula', 'Appalachians and Cumberland Plateau', 'Ozarks Complex', 'Northern Plains', 'Central Plains', 'Southern Plains', 'Northern Rockies', 'Southern Rockies and Colorado Plateau', 'Desert Southwest', 'Great Basin', 'Pacific Northwest', 'Pacific Southwest']

# asymmetric_err = [pdf_results_norm['mean_diff_mid'] - pdf_results_norm['mean_diff_low'], pdf_results_norm['mean_diff_high'] - pdf_results_norm['mean_diff_mid']]
# plt.bar(np.arange(1, 18, 1), pdf_results_norm['mean_diff_mid'], yerr=np.array(asymmetric_err),
#         facecolor=matplotlib.colors.to_rgba('royalblue', 0.67), edgecolor='black')
# plt.xticks(np.arange(1, 18, 1))
# plt.xticks([])
# # plt.xticks(rotation=90)
# # plt.ylim([-12, 12])
# plt.yticks(np.arange(-5, 15, 5), fontsize=14)
# plt.axhline(color='k')
# plt.ylabel('change in mean (%)', fontsize=14)
# plt.title('Change in Mean', fontsize=18)
# # plt.tight_layout()
# plt.show()

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


# asymmetric_err = [pdf_results_norm['std_diff_mid'] - pdf_results_norm['std_diff_low'], pdf_results_norm['std_diff_high'] - pdf_results_norm['std_diff_mid']]
# plt.bar(np.arange(1, 18, 1), pdf_results_norm['std_diff_mid'], yerr=np.array(asymmetric_err),
#         facecolor=matplotlib.colors.to_rgba('firebrick', 0.67), edgecolor='black')
# plt.xticks(np.arange(1, 18, 1))
# plt.xticks([])
# # plt.xticks(rotation=90)
# # plt.ylim([-12, 12])
# plt.yticks(np.arange(0, 50, 10), fontsize=14)
# plt.axhline(color='k')
# plt.ylabel('change in standard deviation (%)', fontsize=14)
# plt.title('Change in Standard Deviation', fontsize=18)
# # plt.tight_layout()
# plt.show()


fig, ax = plt.subplots(figsize=(8, 12))
asymmetric_err = [pdf_results_norm['std_diff_mid'] - pdf_results_norm['std_diff_low'], pdf_results_norm['std_diff_high'] - pdf_results_norm['std_diff_mid']]
ax.barh(np.arange(1, 18, 1), pdf_results_norm['std_diff_mid'], xerr=np.array(asymmetric_err),
        facecolor=matplotlib.colors.to_rgba('firebrick', 0.67), edgecolor='black')
ax.invert_yaxis()
plt.yticks(np.arange(1, 18, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(0, 50, 10), fontsize=20)
plt.xlim([0, 45])
ax.axvline(color='k')
plt.xlabel('change in standard deviation (%)', fontsize=20)
plt.title('Change in Standard Deviation', fontsize=24)
plt.show()



#%% Figure 2: Graphically looking at changes in intensity (also Supporting Info Figures S2-S17)
domain = 5  # using Great Lakes as an example
domain_names = ['Northeast', 'Mid Atlantic', 'Southeast', 'Atlantic Neotropical', 'Great Lakes', 'Prairie Peninsula', 'Appalachians and Cumberland Plateau', 'Ozarks Complex', 'Northern Plains', 'Central Plains', 'Southern Plains', 'Northern Rockies', 'Southern Rockies and Colorado Plateau', 'Desert Southwest', 'Great Basin', 'Pacific Northwest', 'Pacific Southwest']
domain_name = domain_names[domain-1]
pdf_1 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_first_half_pdf.npy')
pdf_1 = np.sort(pdf_1[~np.isnan(pdf_1)])
if domain in [1, 5, 7, 10]:
    pdf_1 = np.sort(pdf_1)[:-1]
pdf_2 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_second_half_pdf.npy')
pdf_2 = np.sort(pdf_2[~np.isnan(pdf_2)])
pdf_1_both = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_both_first_half_pdf.npy')
pdf_1_both = np.sort(pdf_1_both[~np.isnan(pdf_1_both)])
if domain in [1, 5, 7, 10]:
    pdf_1_both = np.sort(pdf_1_both)[:-1]
pdf_2_both = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_both_second_half_pdf.npy')
pdf_2_both = np.sort(pdf_2_both[~np.isnan(pdf_2_both)])
per_99 = np.round(np.quantile(pdf_1_both, q=0.99), 1)
max_99 = np.max([pdf_1[-1], pdf_2_both[-1]])+0.1

both = True
if both:
    pdf_1 = pdf_1_both
    pdf_2 = pdf_2_both
num_bins = 24  # put in one fewer than what you really want
bin_size_low = np.linspace(1, per_99, num_bins)[1] - np.linspace(1, per_99, num_bins)[0]
bin_size_99 = (max_99-per_99)/num_bins
bins_99_up = np.arange(per_99, max_99+bin_size_99, bin_size_99)
bins_99_down = np.arange(per_99, -bin_size_99, -bin_size_99)
bin_array = np.append(np.flip(bins_99_down), bins_99_up)
bin_array = np.unique(bin_array)
pdf_1_density = np.histogram(pdf_1, bins=np.linspace(1, per_99, num_bins), density=True)
pdf_2_density = np.histogram(pdf_2, bins=np.linspace(1, per_99, num_bins), density=True)
pdf_1_99 = np.histogram(pdf_1, bins=bin_array, density=True)
pdf_2_99 = np.histogram(pdf_2, bins=bin_array, density=True)

pdf_1_hist_density = np.histogram(pdf_1_both, bins=np.linspace(1, per_99, num_bins), density='True')
pdf_2_hist_density = np.histogram(pdf_2_both, bins=np.linspace(1, per_99, num_bins), density='True')
hist_diff_density = pdf_2_hist_density[0]-pdf_1_hist_density[0]
hist_diff_density[hist_diff_density==np.inf] = pdf_2_hist_density[0][hist_diff_density==np.inf]
hist_diff_density[hist_diff_density==-100] = -pdf_1_hist_density[0][hist_diff_density==-100]



# side-by-side histograms of early and late periods
pdf_1_hist_density = np.histogram(pdf_1_both, bins=np.linspace(1, per_99, num_bins), density='True')
pdf_2_hist_density = np.histogram(pdf_2_both, bins=np.linspace(1, per_99, num_bins), density='True')
fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(pdf_1_hist_density[1][:-1]-0.3*bin_size_low/2, pdf_1_hist_density[0]*100, facecolor=matplotlib.colors.to_rgba('firebrick', 0.25), edgecolor='black', width=0.3*bin_size_low)
plt.bar(pdf_1_hist_density[1][:-1]+0.3*bin_size_low/2, pdf_2_hist_density[0]*100, facecolor=matplotlib.colors.to_rgba('royalblue', 0.9), edgecolor='black', width=0.3*bin_size_low)
plt.xticks(fontsize=22)
plt.yticks(np.arange(0, 40, 5), fontsize=22)
plt.ylim([0, np.ceil(np.max(pdf_1_hist_density[0])*100+1)])
plt.ylabel('frequencies (%)', fontsize=24)
plt.xlabel('mm/wet day', fontsize=24)
plt.title('Great Lakes', fontsize=28)
plt.show()

# plot difference in distributions at each bin
fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(pdf_1_hist_density[1][:-1], hist_diff_density*100, facecolor=matplotlib.colors.to_rgba('royalblue', 0.75), edgecolor='black', width=0.8*bin_size_low)
plt.axhline(y=0, color='k')
plt.xticks(fontsize=22)
plt.yticks(np.arange(-0.6, 0.61, 0.2), fontsize=22)
# plt.ylim([0, np.ceil(np.max(pdf_1_hist_density[0])*100+1)])
plt.xlabel('mm/wet day', fontsize=24)
plt.ylabel('frequency change (%)', fontsize=24)
# plt.title('domain ' + str(domain) + ': sub-99th percentile')
# plt.title('Intensity Changes', fontsize=28)
plt.show()

hist_diff_extended = np.append(hist_diff_density/pdf_1_hist_density[0]*100, [np.nan, np.nan])
hist_diff_extended = np.append([np.nan, np.nan], hist_diff_extended)
hist_diff_rolling = pd.Series(hist_diff_extended).rolling(5, min_periods=3).mean()[4:]

# plot normalized difference in distributions at each bin for sub-99th percentiles
fig, ax = plt.subplots(figsize=(14, 8))
plt.bar(pdf_1_hist_density[1][:-1], hist_diff_density/pdf_1_hist_density[0]*100, facecolor=matplotlib.colors.to_rgba('royalblue', 0.75), edgecolor='black', width=0.8*bin_size_low)
plt.plot(pdf_1_hist_density[1][:-1], hist_diff_rolling, 'k', linewidth=5)
plt.axhline(y=0, color='k')
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('mm/wet day', fontsize=24)
plt.ylabel('relative frequency change (%)', fontsize=24)
# plt.title('Relative Intensity Changes', fontsize=28)
plt.show()

# ## plot normalized difference in distributions at each bin for super-99th percentiles
# ind = np.where(pdf_2_99[1]==per_99)[0][0]
# per_density_99_change = 100*(pdf_2_99[0][ind:]-pdf_1_99[0][ind:])/pdf_1_99[0][ind:]
# # plt.bar(bin_array[ind:-1], per_density_99_change, facecolor=matplotlib.colors.to_rgba('royalblue', 0.0), edgecolor='black', width=bin_size_99*0.8, hatch='/')
# plt.bar(bin_array[ind:-1], per_density_99_change, facecolor=matplotlib.colors.to_rgba('royalblue', 0.5), edgecolor='black', width=bin_size_99*0.8)
# plt.bar(bin_array[ind:-1][per_density_99_change==np.inf], 100, facecolor=matplotlib.colors.to_rgba('royalblue', 0.0), edgecolor='black', width=bin_size_99*0.8, hatch='/')
# plt.bar(bin_array[ind:-1][per_density_99_change==-100], -100, facecolor=matplotlib.colors.to_rgba('royalblue', 0.0), edgecolor='black', width=bin_size_99*0.8, hatch='/')
# # per_density_99_change = 100*(pdf_2_99[0][ind:]-pdf_1_99[0][ind:])/pdf_1_99[0][ind:]
# # plt.plot(bin_array[ind:-1], per_density_99_change_rolling, color='royalblue', linewidth=3)
# plt.axhline(y=0, color='k')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('mm/wet day', fontsize=14)
# plt.ylabel('relative frequency change (%)', fontsize=14)
# plt.title(domain_name, fontsize=16)
# plt.show()



#%% Figure 3: Creating stylized versions of each domain's precip trends
for domain in np.arange(1, 18, 1):  # looping over each domain
    pdf_1 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_first_half_pdf.npy')
    pdf_1 = np.sort(pdf_1[~np.isnan(pdf_1)])
    if domain in [1, 5, 7, 10]:
        pdf_1 = np.sort(pdf_1)[:-1]
    pdf_2 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_second_half_pdf.npy')
    pdf_2 = np.sort(pdf_2[~np.isnan(pdf_2)])
    pdf_1_both = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_both_first_half_pdf.npy')
    pdf_1_both = np.sort(pdf_1_both[~np.isnan(pdf_1_both)])
    if domain in [1, 5, 7, 10]:
        pdf_1_both = np.sort(pdf_1_both)[:-1]
    pdf_2_both = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_both_second_half_pdf.npy')
    pdf_2_both = np.sort(pdf_2_both[~np.isnan(pdf_2_both)])
    per_99 = np.round(np.quantile(pdf_1_both, q=0.99),1)
    max_99 = np.max([pdf_1[-1], pdf_2_both[-1]])+0.1

    both = True
    if both:
        pdf_1 = pdf_1_both
        pdf_2 = pdf_2_both
    num_bins = 24  # put in one fewer than what you really want
    bin_size_low = np.linspace(1, per_99, num_bins)[1] - np.linspace(1, per_99, num_bins)[0]
    bin_size_99 = (max_99-per_99)/num_bins
    bins_99_up = np.arange(per_99, max_99+bin_size_99, bin_size_99)
    bins_99_down = np.arange(per_99, -bin_size_99, -bin_size_99)
    bin_array = np.append(np.flip(bins_99_down), bins_99_up)
    bin_array = np.unique(bin_array)
    pdf_1_density = np.histogram(pdf_1, bins=np.linspace(1, per_99, num_bins), density=True)
    pdf_2_density = np.histogram(pdf_2, bins=np.linspace(1, per_99, num_bins), density=True)
    pdf_1_99 = np.histogram(pdf_1, bins=bin_array, density=True)
    pdf_2_99 = np.histogram(pdf_2, bins=bin_array, density=True)

    pdf_1_hist_density = np.histogram(pdf_1_both, bins=np.linspace(1, per_99, num_bins), density='True')
    pdf_2_hist_density = np.histogram(pdf_2_both, bins=np.linspace(1, per_99, num_bins), density='True')
    hist_diff_density = pdf_2_hist_density[0]-pdf_1_hist_density[0]
    hist_diff_density[hist_diff_density==np.inf] = pdf_2_hist_density[0][hist_diff_density==np.inf]
    hist_diff_density[hist_diff_density==-100] = -pdf_1_hist_density[0][hist_diff_density==-100]

    hist_diff_extended = np.append(hist_diff_density/pdf_1_hist_density[0]*100, [np.nan, np.nan])
    hist_diff_extended = np.append([np.nan, np.nan], hist_diff_extended)
    hist_diff_rolling = pd.Series(hist_diff_extended).rolling(5, min_periods=3).mean()[4:]

    pfit = np.polyfit(pdf_1_hist_density[1][:-1], hist_diff_rolling, 4)
    plt.plot(np.linspace(1, per_99, 500), np.polyval(pfit, np.linspace(1, per_99, 500)), 'k', linewidth=3)
    plt.axhline(y=0, color='k')
    plt.xticks([0, per_99], ['0th', '99th'], fontsize=24)
    plt.ylim([-30, 65])
    plt.yticks([-25, 0, 25, 50], fontsize=24)
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



#%% Supporting Information Figure 1: Boundaries of the NEON Domains with Stations and Histogram
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
    s = 25,
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



domain_name = ['Northeast', 'Mid Atlantic', 'Southeast', 'Atlantic Neotropical', 'Great Lakes', 'Prairie Peninsula', 'Appalachians and Cumberland Plateau', 'Ozarks Complex', 'Northern Plains', 'Central Plains', 'Southern Plains', 'Northern Rockies', 'Southern Rockies and Colorado Plateau', 'Desert Southwest', 'Great Basin', 'Pacific Northwest', 'Pacific Southwest']

fig, ax = plt.subplots(figsize=(17, 5))

width = 0.6
ax.bar(domain_name, both_count, width, color='black', edgecolor='black', alpha=0.9)
ax.bar(domain_name, early_count, width, bottom=both_count, color='mediumseagreen', edgecolor='black', alpha=0.9)
ax.bar(domain_name, late_count, width, bottom=both_count+early_count, color='cornflowerblue', edgecolor='black', alpha=0.9)

ax.set_ylabel('Station Count', fontsize=20)
plt.yticks(np.arange(0,125, 25), fontsize=16)
# ax.set_title('Scores by group and gender')
# ax.axes.xaxis.set_visible(False)
ax.axes.xaxis.set_ticklabels([])
plt.xticks(rotation = 30)
# plt.tight_layout()
plt.show()


# # plotting histogram of number of records within each domain
# ghcn_stations_with_domain = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/supplemental_files/US_stations_with_NEON.csv')
# US_ghcn_stations_with_domain = ghcn_stations_with_domain[ghcn_stations_with_domain['station_id'].str.slice(0,2) == 'US']  # US only
# qual_stations_with_domain = US_ghcn_stations_with_domain[(US_ghcn_stations_with_domain['num_qual_years'] > num_qual_years) & (US_ghcn_stations_with_domain['pct_qual_years'] >= percent_qualifying_years)]  # checking for completeness of record
#
# # tweaking domain name list to get y-tick labels
# domain_names = neon_gdf['DomainName']
# domain_names = domain_names.drop_duplicates()
#
# fig, ax = plt.subplots(figsize = (10,8))
# plt.hist(qual_stations_with_domain['NEON_domain'], bins = np.arange(0.5,18.5,1), orientation = 'horizontal', facecolor = matplotlib.colors.to_rgba('black', 0.05), edgecolor = 'black')
# ax.invert_yaxis()
# ax.tick_params(
#     axis='y',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# # plt.yticks(np.arange(1, 18, 1), domain_names[0:17])
# plt.yticks([])
# plt.xlabel('Number of Qualifying Stations')
# # plt.ylabel('NEON Domain')
# plt.show()


stations_with_domain = pd.read_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/NEON_domain_daily_precip_stats.csv')
domain_station_counts = stations_with_domain[stations_with_domain['early_late']==3].groupby('station_domain').count()
domain_station_counts = domain_station_counts['both_days']

fig, ax = plt.subplots(figsize=(8, 12))
ax.barh(np.arange(1, 18, 1), domain_station_counts[1:17],
        facecolor=matplotlib.colors.to_rgba('royalblue', 0.67), edgecolor='black')
ax.invert_yaxis()
# ax.invert_xaxis()
plt.yticks(np.arange(1, 18, 1))
# plt.yticks([])
ax.axes.yaxis.set_ticklabels([])
plt.xticks(np.arange(0, 100, 20), fontsize=20)
ax.axvline(color='k')
plt.xlabel('count', fontsize=20)
plt.title('Stations per Region', fontsize=24)
plt.show()
