
import pandas as pd
import numpy as np
from warnings import warn

def check_missing(data ,output_path=None):
    """
    check the total number & percentage of missing values
    per variable of a pandas Dataframe
    """

    result = pd.concat([data.isnull().sum() ,data.isnull().mean()] ,axis=1)
    result = result.rename(index=str ,columns={0 :'total missing' ,1 :'proportion'})
    if output_path is not None:
        result.to_csv(output_path +'missing.csv')
        print('result saved at', output_path, 'missing.csv')
    return result


def drop_missing(data, axis=0):
    """
    Listwise deletion:
    excluding all cases (listwise) that have missing values
    Parameters
    ----------
    axis: drop cases(0)/columns(1),default 0
    Returns
    -------
    Pandas dataframe with missing cases/columns dropped
    """

    data_copy = data.copy(deep=True)
    data_copy = data_copy.dropna(axis=axis, inplace=False)
    return data_copy


def add_var_denote_NA(data, NA_col=[]):
    """
    creating an additional variable indicating whether the data
    was missing for that observation (1) or not (0).
    """

    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            data_copy[i + '_is_NA'] = np.where(data_copy[i].isnull(), 1, 0)
        else:
            warn("Column %s has no missing cases" % i)

    return data_copy


def impute_NA_with_avg(data, strategy='mean', NA_col=[]):
    """
    replacing the NA with mean/median/most frequent values of that variable.
    Note it should only be performed over training set and then propagated to test set.
    """

    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            if strategy == 'mean':
                data_copy[i] = data_copy[i].fillna(data[i].mean())
            elif strategy == 'median':
                data_copy[i] = data_copy[i].fillna(data[i].median())
            elif strategy == 'mode':
                data_copy[i] = data_copy[i].fillna(data[i].mode()[0])
        else:
            warn("Column %s has no missing" % i)
    return data_copy