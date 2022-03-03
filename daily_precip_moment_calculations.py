#%% Updating Distribution Bootstrapping for Early/Late/Both
import pandas as pd
import numpy as np
import scipy.stats as ss
import time


def pdf_moments_diff(pdf1, pdf2, bootstrap_num, confidence_interval=0.05):
    # calculates the mean, median, standard deviation, skew, and kurtosis for two specified distributions (early and
    # late periods). Also bootstraps the initial distributions to determine uncertainty interval.
    pdf1_len = len(pdf1)
    pdf2_len = len(pdf2)
    pdf1_mean = np.zeros([bootstrap_num])
    pdf1_median = np.zeros([bootstrap_num])
    pdf1_std = np.zeros([bootstrap_num])
    pdf1_skew = np.zeros([bootstrap_num])
    pdf1_kurt = np.zeros([bootstrap_num])
    pdf2_mean = np.zeros([bootstrap_num])
    pdf2_median = np.zeros([bootstrap_num])
    pdf2_std = np.zeros([bootstrap_num])
    pdf2_skew = np.zeros([bootstrap_num])
    pdf2_kurt = np.zeros([bootstrap_num])
    for i in np.arange(0,bootstrap_num,1):
        sample_pdf1 = np.random.choice(pdf1, size = pdf1_len, replace = True)
        pdf1_mean[i] = sample_pdf1.mean()
        pdf1_median[i] = np.median(sample_pdf1)
        pdf1_std[i] = sample_pdf1.std()
        pdf1_skew[i] = ss.skew(sample_pdf1)
        pdf1_kurt[i] = ss.kurtosis(sample_pdf1)
        sample_pdf2 = np.random.choice(pdf2, size = pdf2_len, replace = True)
        pdf2_mean[i] = sample_pdf2.mean()
        pdf2_median[i] = np.median(sample_pdf2)
        pdf2_std[i] = sample_pdf2.std()
        pdf2_skew[i] = ss.skew(sample_pdf2)
        pdf2_kurt[i] = ss.kurtosis(sample_pdf2)
    mean_diff = pdf2_mean - pdf1_mean
    median_diff = pdf2_median - pdf1_median
    std_diff = pdf2_std - pdf1_mean
    skew_diff = pdf2_skew - pdf1_skew
    kurt_diff = pdf2_kurt - pdf1_kurt
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
    return low_mean, median_mean, high_mean, low_median, median_median, high_median, low_std, median_std, high_std, low_skew, median_skew, high_skew, low_kurt, median_kurt, high_kurt


array_size = [17,1]

# preallocating variables
ks_test_p = np.zeros(array_size); ks_test_p[:] = np.nan
ad_test_p = np.zeros(array_size); ad_test_p[:] = np.nan

pdf_1_mean = np.zeros(array_size); pdf_1_mean[:] = np.nan
pdf_1_median = np.zeros(array_size); pdf_1_median[:] = np.nan
pdf_1_std = np.zeros(array_size); pdf_1_std[:] = np.nan
pdf_1_skew = np.zeros(array_size); pdf_1_skew[:] = np.nan
pdf_1_kurt = np.zeros(array_size); pdf_1_kurt[:] = np.nan

pdf_2_mean = np.zeros(array_size); pdf_2_mean[:] = np.nan
pdf_2_median = np.zeros(array_size); pdf_2_median[:] = np.nan
pdf_2_std = np.zeros(array_size); pdf_2_std[:] = np.nan
pdf_2_skew = np.zeros(array_size); pdf_2_skew[:] = np.nan
pdf_2_kurt = np.zeros(array_size); pdf_2_kurt[:] = np.nan

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

both = True  # True: use only stations which entirely span both time periods
for domain in np.arange(1, 18, 1):
    start = time.time()
    if both is True:
        pdf_1 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_both_first_half_pdf.npy')
        pdf_1 = np.sort(pdf_1[~np.isnan(pdf_1)])
        if domain in [1, 5, 7, 10]:  # removing four outlying points
            pdf_1 = np.sort(pdf_1)[:-1]
        pdf_2 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_both_second_half_pdf.npy')
        pdf_2 = np.sort(pdf_2[~np.isnan(pdf_2)])
    elif both is False:
        pdf_1 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_first_half_pdf.npy')
        pdf_1 = np.sort(pdf_1[~np.isnan(pdf_1)])
        if domain in [1, 5, 7, 10]: # removing four outlying points
            pdf_1 = np.sort(pdf_1)[:-1]
        pdf_2 = np.load('/Users/ryanharp/Documents/great_lakes_precip_variability/cleaned_data/domain_' + str(domain) + '_daily_second_half_pdf.npy')
        pdf_2 = np.sort(pdf_2[~np.isnan(pdf_2)])

    # performing Kolmogorov-Smirnov and Anderson-Darling tests
    ks_test_p[domain-1] = ss.ks_2samp(pdf_1, pdf_2)[1]
    ad_test_p[domain-1] = ss.anderson_ksamp([pdf_1, pdf_2])[2]

    pdf_1_mean[domain-1] = np.mean(pdf_1)
    pdf_1_median[domain-1] = np.median(pdf_1)
    pdf_1_std[domain-1] = np.std(pdf_1)
    pdf_1_skew[domain-1] = ss.skew(pdf_1)
    pdf_1_kurt[domain-1] = ss.kurtosis(pdf_1)

    pdf_2_mean[domain-1] = np.mean(pdf_2)
    pdf_2_median[domain-1] = np.median(pdf_2)
    pdf_2_std[domain-1] = np.std(pdf_2)
    pdf_2_skew[domain-1] = ss.skew(pdf_2)
    pdf_2_kurt[domain-1] = ss.kurtosis(pdf_2)

    mean_diff_low[domain-1], mean_diff_mid[domain-1], mean_diff_high[domain-1], median_diff_low[domain-1], median_diff_mid[domain-1], median_diff_high[domain-1],std_diff_low[domain-1], std_diff_mid[domain-1], std_diff_high[domain-1], skew_diff_low[domain-1], skew_diff_mid[domain-1], skew_diff_high[domain-1], kurt_diff_low[domain-1], kurt_diff_mid[domain-1], kurt_diff_high[domain-1] = pdf_moments_diff(pdf_1, pdf_2, 1000, 0.05)
    print(time.time() - start)

# creating pandas dataframe to store results
pdf_results = pd.DataFrame(np.arange(1,18,1), columns=['regions'])
pdf_results['ks_test_p'] = ks_test_p
pdf_results['ad_test_p'] = ad_test_p
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

if both is True:
    pdf_results.to_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_daily_pdf_moments_both_median.csv')
elif both is False:
    pdf_results.to_csv('/Users/ryanharp/Documents/great_lakes_precip_variability/results/bootstrapped_precipitation_daily_pdf_moments_first_and_second_median.csv')
