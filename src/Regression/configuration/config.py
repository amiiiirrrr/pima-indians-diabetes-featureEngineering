"""
config.py is written for define config parameters
"""
import argparse
import os

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Develope"

class BaseConfig:
    """
    This class set static paths and other configs.
    Args:
    argparse :
    The keys that users assign such as sentence, tagging_model and other statictics paths.
    Returns:
    The configuration dict specify text, statics paths and controller flags.
    """
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.run()

    def run(self):
        """
        The run method is written to define config arguments
        :return: None
        """

        self.parser.add_argument("--dir_dataRaw",
                                 default="../../data/Raw/",
                                 type=str,
                                 help="path dataRaw")
        self.parser.add_argument("--path_dataRaw",
                                 default="../data/Raw/diabetes.csv",
                                 type=str,
                                 help="path dataframe")
        self.parser.add_argument("--path_imbalanced",
                                 default="../data/Raw/imbalanced.csv",
                                 type=str,
                                 help="path dataframe")
        self.parser.add_argument("--LOG_PATH",
                                 default="../model/Logs/log.txt",
                                 type=str)
        self.parser.add_argument("--dir_model",
                                 default="../model/trained/",
                                 type=str)
        self.parser.add_argument("--path_model",
                                 default="../model/trained/pre_trained.pickle",
                                 type=str)
        self.parser.add_argument("--dataExploration",
                                 default=True,
                                 type=str)
        self.parser.add_argument("--imbalanceData",
                                 default=True,
                                 type=str)
        self.parser.add_argument("--imbalanceFactor",
                                 default=1,
                                 type=float)
        self.parser.add_argument("--typeModel",
                                 default="linear_model",
                                 # default="neighbors",
                                 # default="svm",
                                 # default="tree",
                                 # default="ensemble",
                                 # default="xgboost",
                                 type=str)
        self.parser.add_argument("--output_path_check",
                                 default="../data/checkout/",
                                 type=str)
        self.parser.add_argument("--columns",
                                 default= [
                                        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                                        'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
                                    ]
                                 )
        self.parser.add_argument("--scaling_method",
                                 # default="robust",
                                 default="minmax",
                                 type=str)
        self.parser.add_argument("--transformation_method",
                                 # default="logarithmic",
                                 default="exponential",
                                 type=str)
        self.parser.add_argument("--featureSelection_method",
                                 # default="recursive_feature_addition",
                                 default="recursive_feature_elimination",
                                 type=str)

    def get_args(self):
        """
        The get_args method is written to return config arguments
        :return: argparse
        """
        return self.parser.parse_args()
