
import numpy as np

def log_transform(data, cols=[]):
    """
    Logarithmic transformation
    """

    data_plusTransform = data.copy(deep=True)
    Transformed = data.copy(deep=True)
    for i in cols:
        if i != 'Outcome':
            data_plusTransform[i + '_log'] = np.log(data[i] + 1)
            Transformed[i] = np.round(np.log(data[i] + 1), 8)
        else:
            data_plusTransform[i + '_log'] = data[i]
            Transformed[i] = data[i]

    return data_plusTransform, Transformed


def exp_transform(data, coef, cols=[]):
    """
    exp transformation
    """

    data_plusTransform = data.copy(deep=True)
    Transformed = data.copy(deep=True)
    for i in cols:
        if i != 'Outcome':
            data_plusTransform[i + '_exp'] = (data[i]) ** coef
            Transformed[i] = np.round(np.log(data[i] + 1), 8)
        else:
            data_plusTransform[i + '_log'] = data[i]
            Transformed[i] = data[i]

    return data_plusTransform, Transformed

class FeatureTransformation:
    def __init__(self, args, data):
        """
        transforms features by scaling each feature to a given range. Default to [0,1].
        X_scaled = (X - X.min / (X.max - X.min)
        :param column: Dataframe
        :return: scaling objects
        """
        self.args = args
        self.data = data

        if self.args.transformation_method=='logarithmic':
            self.data_plusTransform, self.Transformed_data = self.logaTransform()
        if self.args.transformation_method=='exponential':
            self.data_plusTransform, self.Transformed_data = self.expTransform()

    def logaTransform(self):
        """
        transforms features by scaling each feature to a given range. Default to [0,1].
        X_scaled = (X - X.min / (X.max - X.min)
        :param column: a feature column
        :return: a feature column
        """
        data_plusTransform, Transformed = log_transform(data=self.data, cols=self.args.columns)
        return data_plusTransform, Transformed


    def expTransform(self):
        """
        removes the median and scales the data according to the quantile range (defaults to IQR)
        X_scaled = (X - X.median) / IQR
        :param column: a feature column
        :return: a feature column
        """
        data_plusTransform, Transformed = exp_transform(data=self.data, cols=self.args.columns, coef=0.2)
        return data_plusTransform, Transformed