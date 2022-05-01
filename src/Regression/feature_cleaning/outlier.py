import pandas as pd
import numpy as np


# from warnings import warn

# 2018.11.07 Created by Eamon.Zhang


def outlier_detect_MAD(data, col, threshold=3.5):
    """
    outlier detection by Median and Median Absolute Deviation Method (MAD)
    The median of the residuals is calculated. Then, the difference is calculated between each historical value and this median.
    These differences are expressed as their absolute values, and a new median is calculated and multiplied by
    an empirically derived constant to yield the median absolute deviation (MAD).
    If a value is a certain number of MAD away from the median of the residuals,
    that value is classified as an outlier. The default threshold is 3 MAD.

    This method is generally more effective than the mean and standard deviation method for detecting outliers,
    but it can be too aggressive in classifying values that are not really extremely different.
    Also, if more than 50% of the data points have the same value, MAD is computed to be 0,
    so any value different from the residual median is classified as an outlier.
    """

    median = data[col].median()
    median_absolute_deviation = np.median([np.abs(y - median) for y in data[col]])
    modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in data[col]])
    outlier_index = np.abs(modified_z_scores) > threshold
    print('Num of outlier detected:', outlier_index.value_counts()[1])
    print('Proportion of outlier detected', outlier_index.value_counts()[1] / len(outlier_index))
    return outlier_index


def drop_outlier(data, outlier_index):
    """
    drop the cases that are outliers
    """

    data_copy = data[~outlier_index]
    return data_copy


def impute_outlier_with_avg(data, col, outlier_index, strategy='mean'):
    """
    impute outlier with mean/median/most frequent values of that variable.
    """

    data_copy = data.copy(deep=True)
    if strategy == 'mean':
        data_copy.loc[outlier_index, col] = data_copy[col].mean()
    elif strategy == 'median':
        data_copy.loc[outlier_index, col] = data_copy[col].median()
    elif strategy == 'mode':
        data_copy.loc[outlier_index, col] = data_copy[col].mode()[0]

    return data_copy