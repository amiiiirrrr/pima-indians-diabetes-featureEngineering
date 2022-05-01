"""
dataloader.py is written for loading dataset
"""

import logging
import os
from sklearn.model_selection import train_test_split
from ..utils import load_csv
import matplotlib.pyplot as plt
import csv
import seaborn as sns

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

class DataframeLoader:
    """
    Processor for the dataset
    """
    def __init__(self, args):
        self.args = args                                                 # get arguments from config file
        self.log_file = open(self.args.LOG_PATH, "w")                    # open log file
        if os.path.exists(self.args.path_dataRaw):                       # if dataframe is not created
            logging.info("Dataframe is exist.")                          # yet, DataframeCreation class
        else:                                                            # Dataframe then load csv file
            logging.info("Dataframe is not exist")                       # can be sensible
            self.log_file.write("Dataframe is not exist.")

        # self.use_cols = [
        #     'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
        #     'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
        # ]

        self.raw_data = load_csv(self.args.path_dataRaw)

        if self.args.dataExploration:
            self.show_dataInfo()
            # self.correlation_analysis()

    def show_dataInfo(self):
        """
        show data information
        :param data_df: csv
        """
        self.raw_data.head()
        self.raw_data.info()
        self.raw_data.describe().T

    def correlation_analysis(self):
        """
        show data correlation
        :param data_df: csv
        """
        plt.figure(figsize=[12, 8])
        sns.heatmap(self.raw_data.corr(), annot=True, cmap='RdYlGn', vmin=-1, vmax=1, center=0)
        plt.show()

    def count_variables(self):
        """
        count of variables of outcome
        :param data_df: csv
        """
        return self.raw_data.Outcome.value_counts()

    def make_imbalance(self):
        """
        imbalance data
        :param data_df: csv
        :return data_df_imbalanced: csv
        """
        (negative, positive) = self.count_variables()
        print("positive", positive)
        print("negative", negative)
        number_positives = positive * self.args.imbalanceFactor
        with open(self.args.path_dataRaw, 'r') as inp, open(self.args.path_imbalanced, 'w') as out:
            writer = csv.writer(out)
            counter = 0
            for row in csv.reader(inp):
                if row[-1] == "1":
                    if counter > number_positives:
                        continue
                    counter += 1
                writer.writerow(row)

        return load_csv(self.args.path_imbalanced)

    def split_dataset(self, data):
        """
        spilt dataset to train and test sets.
        :param raw: array
        :return: X_train, x_test, Y_train, y_test
        """

        x_train, x_test, y_train, y_test = train_test_split(
                        data.drop(labels=['Outcome'], axis=1),
                        data.Outcome, test_size=0.2,
                        random_state=0
        )                                                                    # Split our data by 80%
                                                                             # training and 20% testing
        return x_train, x_test, y_train, y_test

