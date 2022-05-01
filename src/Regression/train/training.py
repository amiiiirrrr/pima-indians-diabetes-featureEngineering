"""
training.py is written for training model
"""

import logging

__author__ = "Amir Mousavi"
__license__ = "Public Domain"
__version__ = "1.0.0"
__maintainer__ = "Amir Mousavi"
__email__ = "azmusavi19@gmail.com"
__status__ = "Production"

logger = logging.getLogger(__name__)

class TrainingEvaluation:
    """
    Train the model
    """
    def __init__(self, args, init_model):
        self.args = args
        self.log_file = open(self.args.LOG_PATH, "w")
        self.init_model = init_model

    def train(self, x_train, y_train):
        """
        The train method is written to train the model
        :param x_train: list
        :param y_train: list
        :return: model
        """
        logger.info("***** Running training *****")
        logger.info("Num examples = %d", len(x_train))
        # Fit our model
        model = self.init_model.fit(x_train, y_train)       # Strart training
        return model   

    def evaluate(self, model, x_test, y_test):
        """
        The evaluate method is written to predict values from the test data
        :param model
        :param x_test: list
        :param y_test: list
        :return: list
        """
        #
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(x_test))
        return model.score(x_test, y_test)