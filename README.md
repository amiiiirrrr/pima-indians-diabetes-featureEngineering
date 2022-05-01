# pima-indians-diabetes-featureEngineering
pima-indians-diabetes with feature cleaning, feature engineering and feature selection methods

## Structure

### make dataset imbalance
making dataset imbalance will able you to evaluate your machine learning methods and data engineering methods under harder conditions. Although, you can skip this stage by modifying config.py .

### Data cleaning 
#### Missing value checking is  another stage that you can do it to see which samples is missing.
#### Detect outlier data: outlier detection by Median and Median Absolute Deviation Method (MAD). replacing the outlier by mean/median/most frequent values of that variable

### feature_engineering section

#### Feature Scaling

minmaxScaler: transforms features by scaling each feature to a given range. Default to [0,1] X_scaled = (X - X.min / (X.max - X.min)

robustScaler: removes the median and scales the data according to the quantile range (defaults to IQR) X_scaled = (X - X.median) / IQR

#### Feature Transformation

Logarithmic transformation

exponential transformation

### feature_selection section

#### recursive feature elimination

recursive feature elimination with RandomForest with the method same as the guide

1.Rank the features according to their importance derived from a machine learning algorithm: it can be tree importance, or LASSO / Ridge, or the linear / logistic regression coefficients.
2.Remove one feature -the least important- and build a machine learning algorithm utilizing the remaining features.
3.Calculate a performance metric of your choice: roc-auc, mse, rmse, accuracy.
4.If the metric decreases by more of an arbitrarily set threshold, then that feature is important and should be kept. Otherwise, we can remove that feature.
5.Repeat steps 2-4 until all features have been removed (and therefore evaluated) and the drop in performance assessed.

#### Recursive Feature Addition (with Random Forests Importance)

recursive feature addition with RandomForest with the method same as the guide

1.Rank the features according to their importance derived from a machine learning algorithm: it can be tree importance, or LASSO / Ridge, or the linear / logistic regression coefficients.
2.Build a machine learning model with only 1 feature, the most important one, and calculate the model metric for performance.
3.Add one feature -the most important- and build a machine learning algorithm utilizing the added and any feature from previous rounds.
4.Calculate a performance metric of your choice: roc-auc, mse, rmse, accuracy.
5.If the metric increases by more than an arbitrarily set threshold, then that feature is important and should be kept. Otherwise, we can remove that feature.
6.Repeat steps 2-5 until all features have been removed (and therefore evaluated) and the drop in performance assessed.
