"""
helpers.py is written to provide helper functions
"""

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

import os
import pandas as pd

def make_directory(path):
    """
    make directory if not exist
    :param path: str
    :return: 
        None
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        # directory already exists
        pass

def load_csv(path):
    """
    make directory if not exist
    :param path: str
    :return: csv file
    """

    return pd.read_csv(path)
