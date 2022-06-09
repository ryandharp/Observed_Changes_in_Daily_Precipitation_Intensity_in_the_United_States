import multiprocessing as mp
import pandas as pd
import numpy as np
import scipy.stats as ss
import time
import fnmatch
import os
import os.path

def bootstrap_pdf_creator(domain_num):

    # looping over first half years
    domain_first_half_pdf = np.array([])

    for yr in np.arange(1951, 1981, 2):
        # fpath = '/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/bootstrapping_blocks/domain_bootstrapping/domain_' + str(domain_num) + '/first_half/' + str(yr) + '_' + str(yr+1) + '/'
        fpath = '/home/rdh0715/domain_bootstrapping/domain_' + str(domain_num) + '/first_half/' + str(yr) + '_' + str(yr+1) + '/'
        block_file_list = fnmatch.filter(os.listdir(fpath), '*.npy')
        all_blocks = np.array([np.load(fpath + fname) for fname in block_file_list])
        num_blocks = len(block_file_list)
        rand_block_nums = np.random.choice(np.arange(num_blocks), size = num_blocks, replace = True)
        for block_num in rand_block_nums:
            domain_first_half_pdf = np.concatenate((domain_first_half_pdf, all_blocks[block_num]))


    # looping over second half years
    domain_second_half_pdf = np.array([])

    for yr in np.arange(1991, 2021, 2):
        # fpath = '/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/bootstrapping_blocks/domain_bootstrapping/domain_' + str(domain_num) + '/second_half/' + str(yr) + '_' + str(yr+1) + '/'
        fpath = '/home/rdh0715/domain_bootstrapping/domain_' + str(domain_num) + '/second_half/' + str(yr) + '_' + str(yr+1) + '/'
        block_file_list = fnmatch.filter(os.listdir(fpath), '*.npy')
        all_blocks = np.array([np.load(fpath + fname) for fname in block_file_list])
        num_blocks = len(block_file_list)
        rand_block_nums = np.random.choice(np.arange(num_blocks), size = num_blocks, replace = True)
        for block_num in rand_block_nums:
            domain_second_half_pdf = np.concatenate((domain_second_half_pdf, all_blocks[block_num]))


    # clearing out nans for future calculations
    domain_first_half_pdf = domain_first_half_pdf[~np.isnan(domain_first_half_pdf)]
    domain_second_half_pdf = domain_second_half_pdf[~np.isnan(domain_second_half_pdf)]

    return domain_first_half_pdf, domain_second_half_pdf


def pdf_moments_diff(bootstrap_num, domain_num, confidence_interval=0.05):
    # calculates the mean, median, standard deviation, skew, and kurtosis for two specified distributions (early and
    # late periods). Also bootstraps the initial distributions to determine uncertainty interval.

    pdf1_mean = np.zeros([bootstrap_num])
    pdf1_median = np.zeros([bootstrap_num])
    pdf1_std = np.zeros([bootstrap_num])
    pdf1_skew = np.zeros([bootstrap_num])
    pdf1_kurt = np.zeros([bootstrap_num])
    pdf1_99 = np.zeros([bootstrap_num])
    pdf2_mean = np.zeros([bootstrap_num])
    pdf2_median = np.zeros([bootstrap_num])
    pdf2_std = np.zeros([bootstrap_num])
    pdf2_skew = np.zeros([bootstrap_num])
    pdf2_kurt = np.zeros([bootstrap_num])

    num_bins = 25  # put in one fewer than what you really want
    pdf_shift = np.zeros([bootstrap_num, num_bins-1])
    pdf_shift_quantiles = np.zeros([3, num_bins-1])
    pdf_shift_percentiles = np.zeros([bootstrap_num, 20])
    pdf_shift_percentiles_quantiles = np.zeros([3, 20])
    pdf_shift_percentiles_99 = np.zeros([bootstrap_num, 20])
    pdf_shift_percentiles_99_quantiles = np.zeros([3, 20])

    start = time.time()
    for i in np.arange(0, bootstrap_num, 1):
        if i % 10 == 0:
            print('domain ' +str(domain_num) + ': ' + str(i) + ' at ' + str(time.time() - start) + ' seconds')
        if i == (bootstrap_num - 1):
            print('domain ' + str(domain_num) + ': final loop at ' + str(time.time() - start) + ' seconds')
        pdf1, pdf2 = bootstrap_pdf_creator(domain_num)
        pdf1_mean[i] = pdf1.mean()
        pdf1_median[i] = np.median(pdf1)
        pdf1_std[i] = pdf1.std()
        pdf1_skew[i] = ss.skew(pdf1)
        pdf1_kurt[i] = ss.kurtosis(pdf1)
        pdf2_mean[i] = pdf2.mean()
        pdf2_median[i] = np.median(pdf2)
        pdf2_std[i] = pdf2.std()
        pdf2_skew[i] = ss.skew(pdf2)
        pdf2_kurt[i] = ss.kurtosis(pdf2)

        per_99 = np.round(np.quantile(pdf1, q=0.99), 1)
        pdf1_99[i] = per_99
        pdf_1_hist_density = np.histogram(pdf1, bins=np.linspace(1, per_99, num_bins))
        pdf_2_hist_density = np.histogram(pdf2, bins=np.linspace(1, per_99, num_bins))
        hist_diff_density = pdf_2_hist_density[0] - pdf_1_hist_density[0]
        hist_diff_density[hist_diff_density == np.inf] = pdf_2_hist_density[0][hist_diff_density == np.inf]
        hist_diff_density[hist_diff_density == -100] = -pdf_1_hist_density[0][hist_diff_density == -100]
        pdf_shift[i, :] = hist_diff_density / pdf_1_hist_density[0] * 100

        percentile_bins = np.array([])
        for b in np.arange(0, 1.01, 0.05):
            percentile_bins = np.append(percentile_bins, np.round(np.quantile(pdf1, q=b), 1))
        percentile_bins[-1] = np.inf
        percentile_bins = np.sort(percentile_bins)
        pdf_1_hist_density_percentiles = np.histogram(pdf1, bins=percentile_bins)
        pdf_2_hist_density_percentiles = np.histogram(pdf2, bins=percentile_bins)
        pdf_1_hist_density_percentiles_norm = pdf_1_hist_density_percentiles[0]/np.nansum(pdf_1_hist_density_percentiles[0])
        pdf_2_hist_density_percentiles_norm = pdf_2_hist_density_percentiles[0]/np.nansum(pdf_2_hist_density_percentiles[0])
        pdf_shift_percentiles[i, :] = ((pdf_2_hist_density_percentiles_norm - pdf_1_hist_density_percentiles_norm) / pdf_1_hist_density_percentiles_norm) * 100

        percentile_bins = np.array([])
        for b in np.arange(0, 0.99, 0.1):
            percentile_bins = np.append(percentile_bins, np.round(np.quantile(pdf1, q=b), 1))
        for b in np.arange(0.99, 1.0001, 0.0005):
            percentile_bins = np.append(percentile_bins, np.round(np.quantile(pdf1, q=b), 1))
        percentile_bins[-1] = np.inf
        percentile_bins = np.sort(percentile_bins)
        pdf_1_hist_density_percentiles_99 = np.histogram(pdf1, bins=percentile_bins)
        pdf_1_hist_density_percentiles_99_norm = pdf_1_hist_density_percentiles_99[0] / np.nansum(pdf_1_hist_density_percentiles_99[0])
        pdf_2_hist_density_percentiles_99 = np.histogram(pdf2, bins=percentile_bins)
        pdf_2_hist_density_percentiles_99_norm = pdf_2_hist_density_percentiles_99[0] / np.nansum(pdf_2_hist_density_percentiles_99[0])
        pdf_shift_percentiles_99[i, :] = ((pdf_2_hist_density_percentiles_99_norm[-20:] - pdf_1_hist_density_percentiles_99_norm[-20:]) / pdf_1_hist_density_percentiles_99_norm[-20:]) * 100

    mean_diff = pdf2_mean - pdf1_mean
    median_diff = pdf2_median - pdf1_median
    std_diff = pdf2_std - pdf1_std
    skew_diff = pdf2_skew - pdf1_skew
    kurt_diff = pdf2_kurt - pdf1_kurt
    init_mean = np.quantile(pdf1_mean, q=0.5)
    init_median = np.quantile(pdf1_median, q=0.5)
    init_std = np.quantile(pdf1_std, q=0.5)
    init_skew = np.quantile(pdf1_skew, q=0.5)
    init_kurt = np.quantile(pdf1_kurt, q=0.5)
    low_mean = np.quantile(mean_diff, q=confidence_interval/2)
    median_mean = np.quantile(mean_diff, q=0.5)
    high_mean = np.quantile(mean_diff, q=1-confidence_interval/2)
    low_median = np.quantile(median_diff, q=confidence_interval/2)
    median_median = np.quantile(median_diff, q=0.5)
    high_median = np.quantile(median_diff, q=1-confidence_interval/2)
    low_std = np.quantile(std_diff, q=confidence_interval/2)
    median_std = np.quantile(std_diff, q=0.5)
    high_std = np.quantile(std_diff, q=1-confidence_interval/2)
    low_skew = np.quantile(skew_diff, q=confidence_interval/2)
    median_skew = np.quantile(skew_diff, q=0.5)
    high_skew = np.quantile(skew_diff, q=1-confidence_interval/2)
    low_kurt = np.quantile(kurt_diff, q=confidence_interval/2)
    median_kurt = np.quantile(kurt_diff, q=0.5)
    high_kurt = np.quantile(kurt_diff, q=1-confidence_interval/2)

    # np.save('/home/rdh0715/block_bootstrap_results/per_99/per_99_domain_' + str(domain_num) + '.npy', pdf1_99)

    # pdf_shift_quantiles[0, :] = np.nanquantile(pdf_shift, axis=0, q=0.05)
    # pdf_shift_quantiles[1, :] = np.nanquantile(pdf_shift, axis=0, q=0.5)
    # pdf_shift_quantiles[2, :] = np.nanquantile(pdf_shift, axis=0, q=0.95)
    # np.save('/home/rdh0715/block_bootstrap_results/pdf_shift/pdf_shift_domain_' + str(domain_num) + '.npy', pdf_shift_quantiles)

    pdf_shift_percentiles_quantiles[0, :] = np.nanquantile(pdf_shift_percentiles, axis=0, q=0.05)
    pdf_shift_percentiles_quantiles[1, :] = np.nanquantile(pdf_shift_percentiles, axis=0, q=0.5)
    pdf_shift_percentiles_quantiles[2, :] = np.nanquantile(pdf_shift_percentiles, axis=0, q=0.95)
    np.save('/home/rdh0715/block_bootstrap_results/pdf_shift/pdf_shift_percentiles_domain_' + str(domain_num) + '.npy', pdf_shift_percentiles_quantiles)

    pdf_shift_percentiles_99_quantiles[0, :] = np.nanquantile(pdf_shift_percentiles_99, axis=0, q=0.05)
    pdf_shift_percentiles_99_quantiles[1, :] = np.nanquantile(pdf_shift_percentiles_99, axis=0, q=0.5)
    pdf_shift_percentiles_99_quantiles[2, :] = np.nanquantile(pdf_shift_percentiles_99, axis=0, q=0.95)
    np.save('/home/rdh0715/block_bootstrap_results/pdf_shift/pdf_shift_percentiles_99_domain_' + str(domain_num) + '.npy', pdf_shift_percentiles_99_quantiles)

    return init_mean, init_median, init_std, init_skew, init_kurt, low_mean, median_mean, high_mean, low_median, median_median, high_median, low_std, median_std, high_std, low_skew, median_skew, high_skew, low_kurt, median_kurt, high_kurt


array_size = [20, 1]

pdf_1_mean = np.zeros(array_size); pdf_1_mean[:] = np.nan
pdf_1_median = np.zeros(array_size); pdf_1_median[:] = np.nan
pdf_1_std = np.zeros(array_size); pdf_1_std[:] = np.nan
pdf_1_skew = np.zeros(array_size); pdf_1_skew[:] = np.nan
pdf_1_kurt = np.zeros(array_size); pdf_1_kurt[:] = np.nan

mean_diff_low = np.zeros(array_size); mean_diff_low[:] = np.nan
mean_diff_mid = np.zeros(array_size); mean_diff_mid[:] = np.nan
mean_diff_high = np.zeros(array_size); mean_diff_high[:] = np.nan
median_diff_low = np.zeros(array_size); median_diff_low[:] = np.nan
median_diff_mid = np.zeros(array_size); median_diff_mid[:] = np.nan
median_diff_high = np.zeros(array_size); median_diff_high[:] = np.nan
std_diff_low = np.zeros(array_size); std_diff_low[:] = np.nan
std_diff_mid = np.zeros(array_size); std_diff_mid[:] = np.nan
std_diff_high = np.zeros(array_size); std_diff_high[:] = np.nan
skew_diff_low = np.zeros(array_size); skew_diff_low[:] = np.nan
skew_diff_mid = np.zeros(array_size); skew_diff_mid[:] = np.nan
skew_diff_high = np.zeros(array_size); skew_diff_high[:] = np.nan
kurt_diff_low = np.zeros(array_size); kurt_diff_low[:] = np.nan
kurt_diff_mid = np.zeros(array_size); kurt_diff_mid[:] = np.nan
kurt_diff_high = np.zeros(array_size); kurt_diff_high[:] = np.nan


def wrapper(domain):
    print(domain)
    bootstrap_num = 1000

    pdf_1_mean[domain - 1], pdf_1_median[domain - 1], pdf_1_std[domain - 1], pdf_1_skew[domain - 1], pdf_1_kurt[
        domain - 1], \
    mean_diff_low[domain - 1], mean_diff_mid[domain - 1], mean_diff_high[domain - 1], median_diff_low[domain - 1], \
    median_diff_mid[domain - 1], median_diff_high[domain - 1], std_diff_low[domain - 1], std_diff_mid[domain - 1], \
    std_diff_high[domain - 1], skew_diff_low[domain - 1], skew_diff_mid[domain - 1], skew_diff_high[domain - 1], \
    kurt_diff_low[domain - 1], kurt_diff_mid[domain - 1], kurt_diff_high[domain - 1] = pdf_moments_diff(bootstrap_num, domain,
                                                                                                        0.05)

    # creating pandas dataframe to store results
    pdf_results = pd.DataFrame(np.arange(1, 21, 1), columns=['regions'])
    pdf_results['pdf_1_mean'] = pdf_1_mean
    pdf_results['pdf_1_median'] = pdf_1_median
    pdf_results['pdf_1_std'] = pdf_1_std
    pdf_results['pdf_1_skew'] = pdf_1_skew
    pdf_results['pdf_1_kurt'] = pdf_1_kurt
    pdf_results['mean_diff_low'] = mean_diff_low
    pdf_results['mean_diff_mid'] = mean_diff_mid
    pdf_results['mean_diff_high'] = mean_diff_high
    pdf_results['median_diff_low'] = median_diff_low
    pdf_results['median_diff_mid'] = median_diff_mid
    pdf_results['median_diff_high'] = median_diff_high
    pdf_results['std_diff_low'] = std_diff_low
    pdf_results['std_diff_mid'] = std_diff_mid
    pdf_results['std_diff_high'] = std_diff_high
    pdf_results['skew_diff_low'] = skew_diff_low
    pdf_results['skew_diff_mid'] = skew_diff_mid
    pdf_results['skew_diff_high'] = skew_diff_high
    pdf_results['kurt_diff_low'] = kurt_diff_low
    pdf_results['kurt_diff_mid'] = kurt_diff_mid
    pdf_results['kurt_diff_high'] = kurt_diff_high

    pdf_results.to_csv(
        '/home/rdh0715/block_bootstrap_results/results/domain_bootstrapped_daily_precipitation_pdf_moments_domain_' + str(
            domain) + '.csv')


if __name__ == '__main__':
    pool = mp.Pool(processes=21)
    pool.map(wrapper, np.arange(1, 21, 1))
