"""
load_model.py is written for loading the model
"""

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from ..utils import make_directory

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

def load_trained(path_model):
    """
    Load trained model
    :param path_model: str
    :return: pickle
    """
    with open(path_model, 'rb') as file:        
        pickle_model = pickle.load(file)        # Load model from save path
    return pickle_model

def save_model(model, path_model):
    """
    Save trained model in given path
    :param model
    :param path_model: str
    :return: 
        None
    """
    with open(path_model, 'wb') as file:
        pickle.dump(model, file)

type_model = {
    "linear_model": LogisticRegression,
    "neighbors": KNeighborsClassifier,
    "svm": SVC,
    "tree": DecisionTreeClassifier,
    "ensemble": RandomForestClassifier,
    "xgboost": XGBClassifier,
}
class RegressionModel:
    """
    This class load model.
    Args:
    args: argparse
    Returns: Model
    """
    def __init__(self, args):
        self.args = args
        make_directory(self.args.dir_model)     # make a dir. to save and load the model

    def load_model(self):
        """
        The loadModel method is written to load model
        :return: model
        """
        return type_model[self.args.typeModel]

    def predict(self, x_test, model_):
        """
        The evaluate method is written to predict values from the test data
        :param dataloader: dataset iterator
        :param prefix: str
        :return: dict
        """
        # 
        return model_.predict(x_test)
