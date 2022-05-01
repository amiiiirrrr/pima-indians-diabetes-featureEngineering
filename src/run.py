"""
run.py is written to train and evaluate the model
"""
import os
import logging
import glob
import numpy as np
import pandas as pd
from Regression.configuration import BaseConfig
from Regression.methods import RegressionModel, save_model, load_trained
from Regression.dataloader import DataframeLoader
from Regression.train import TrainingEvaluation, benchmarkModels
from Regression.feature_cleaning import check_missing, drop_missing, add_var_denote_NA, impute_NA_with_avg
from Regression.feature_cleaning import outlier_detect_MAD, drop_outlier, impute_outlier_with_avg
from Regression.feature_engineering import FeatureScaling, FeatureTransformation
from Regression.feature_selection import recursive_feature_elimination_rf, recursive_feature_addition_rf

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

logger = logging.getLogger(__name__)

class ModelRrunner:
    """
    A class to run the training and testing the model
    """
    def __init__(self):

        self.args = BaseConfig().get_args()					# To create config object

        self.model_obj = RegressionModel(self.args)			# To create model object 

        self.dataloader_obj = DataframeLoader(self.args)	# To create dataloader object


        # if (os.path.exists(self.args.dir_model)			    # Check output dir. already exists
        #     and os.listdir(self.args.dir_model)
        # ):
        #     raise ValueError(
        #         "Output directory ({}) already exists and is not empty. "
        #     )

    def run(self):
        """
        A method to run the training and testing the model
        :return: 
            None
        """
        #################################### Make dataset Imbalance ###########################################
        # Make the dataset imbalance with 10% of True data
        if self.args.imbalanceData:
            raw_data = self.dataloader_obj.make_imbalance()

        ##################################### Data cleaning #######################################
        # Missing value checking
        self.missing_value_checking(raw_data)

        label = raw_data.iloc[:, -1]
        # raw_data = raw_data.iloc[:, :-1]

        # Detect outlier data
        # outlier detection by Median and Median Absolute Deviation Method (MAD)
        data_imputed_outliers = raw_data
        for col in self.args.columns:
            if col != 'Outcome':
                try:
                    index = outlier_detect_MAD(data=data_imputed_outliers, col=col, threshold=3.5)

                    # Mean/Median/Mode Imputation
                    # replacing the outlier by mean/median/most frequent values of that variable
                    data_imputed_outliers = impute_outlier_with_avg(data=data_imputed_outliers, col=col,
                                                       outlier_index=index, strategy='mean')
                except:
                    continue

        ############################# feature_engineering section ################################
        # Feature Scaling
        self.FeatureScaling_obj = FeatureScaling(self.args, data_imputed_outliers)
        scalered_data = self.FeatureScaling_obj.Scaler_data
        scalered_data.to_csv("../data/checkout/scalered_data.csv", index=False, header=True)

        # Feature Transformation
        self.FeatureTransform_obj = FeatureTransformation(self.args, scalered_data)
        transformed_data = self.FeatureTransform_obj.Transformed_data
        transformed_data.to_csv("../data/checkout/transformed_data.csv", index=False, header=True)

        # After Feature Transformation suddenly non values has occured.
        # Replacing the NA by mean/median/mode of that variable Mean/Median/Mode Imputation
        transformed_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        transformed_data_imputed = impute_NA_with_avg(data=transformed_data, strategy='median', NA_col=self.args.columns)

        ######### splite dataset into train and test fold
        X_train, X_test, y_train, y_test = self.dataloader_obj.split_dataset(transformed_data_imputed)

        ################################ feature_selection section #####################################
        # Recursive Feature Addition with Random Forests Importance
        if self.args.featureSelection_method=='recursive_feature_addition':
            features_to_keep = recursive_feature_addition_rf(X_train=X_train,
                                                                    y_train=y_train,
                                                                    X_test=X_test,
                                                                    y_test=y_test,
                                                                    tol=0.001)
        if self.args.featureSelection_method=='recursive_feature_elimination':
            features_to_keep = recursive_feature_elimination_rf(X_train=X_train,
                                                                       y_train=y_train,
                                                                       X_test=X_test,
                                                                       y_test=y_test,
                                                                       tol=0.001)
        ########### Now we know which features we want to use to Train ML models
        print("features_to_keep", features_to_keep)

        data_kept = self.select_kept_features(transformed_data_imputed, features_to_keep)

        benchmarkModels(data_kept, label)

    def select_kept_features(self, data, features_to_keep):
        """
        select just those features that have been selected by selection method
        :param data: Dataframe
        :return selected_data: Dataframe
        """
        selected_data = pd.DataFrame(columns = features_to_keep)
        for col in features_to_keep:
            selected_data[col] = data[col]
        return selected_data

    def missing_value_checking(self, data):
        """
        Missing value checking
        :param data: Dataframe
        """
        check_missing(data=data, output_path=self.args.output_path_check)

if __name__ == '__main__':
    run_model_obj = ModelRrunner()
    run_model_obj.run()
