
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler

class FeatureScaling:
    def __init__(self, args, data):
        """
        transforms features by scaling each feature to a given range. Default to [0,1].
        X_scaled = (X - X.min / (X.max - X.min)
        :param column: Dataframe
        :return: scaling objects
        """
        self.args = args
        self.data = data

        if self.args.scaling_method=='minmax':
            self.Scaler_data = self.minmaxScaler()
        if self.args.scaling_method=='robust':
            self.Scaler_data = self.robustScaler()

    def minmaxScaler(self):
        """
        transforms features by scaling each feature to a given range. Default to [0,1].
        X_scaled = (X - X.min / (X.max - X.min)
        :param column: a feature column
        :return: a feature column
        """
        for column in self.args.columns:
            if column != 'Outcome':
                obj = MinMaxScaler().fit(self.data[[column]])
                scalered_column = obj.transform(self.data[[column]])
            else:
                scalered_column = self.data[[column]]

            self.data[column] = scalered_column
        return self.data


    def robustScaler(self):
        """
        removes the median and scales the data according to the quantile range (defaults to IQR)
        X_scaled = (X - X.median) / IQR
        :param column: a feature column
        :return: a feature column
        """
        for column in self.args.columns:
            if column != 'Outcome':
                rs = RobustScaler().fit(self.data[[column]])
                scalered_column = rs.transform(self.data[[column]])
            else:
                scalered_column = self.data[[column]]

            self.data[column] = scalered_column
        return self.data