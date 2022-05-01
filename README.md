# pima-indians-diabetes-featureEngineering
pima-indians-diabetes with feature cleaning, feature engineering and feature selection methods

## Dataset 

1. Title: Pima Indians Diabetes Database

2. Sources:
   (a) Original owners: National Institute of Diabetes and Digestive and
                        Kidney Diseases
   (b) Donor of database: Vincent Sigillito (vgs@aplcen.apl.jhu.edu)
                          Research Center, RMI Group Leader
                          Applied Physics Laboratory
                          The Johns Hopkins University
                          Johns Hopkins Road
                          Laurel, MD 20707
                          (301) 953-6231
   (c) Date received: 9 May 1990

3. Past Usage:
    1. Smith,~J.~W., Everhart,~J.~E., Dickson,~W.~C., Knowler,~W.~C., \&
       Johannes,~R.~S. (1988). Using the ADAP learning algorithm to forecast
       the onset of diabetes mellitus.  In {\it Proceedings of the Symposium
       on Computer Applications and Medical Care} (pp. 261--265).  IEEE
       Computer Society Press.

       The diagnostic, binary-valued variable investigated is whether the
       patient shows signs of diabetes according to World Health Organization
       criteria (i.e., if the 2 hour post-load plasma glucose was at least 
       200 mg/dl at any survey  examination or if found during routine medical
       care).   The population lives near Phoenix, Arizona, USA.

       Results: Their ADAP algorithm makes a real-valued prediction between
       0 and 1.  This was transformed into a binary decision using a cutoff of 
       0.448.  Using 576 training instances, the sensitivity and specificity
       of their algorithm was 76% on the remaining 192 instances.

4. Relevant Information:
      Several constraints were placed on the selection of these instances from
      a larger database.  In particular, all patients here are females at
      least 21 years old of Pima Indian heritage.  ADAP is an adaptive learning
      routine that generates and executes digital analogs of perceptron-like
      devices.  It is a unique algorithm; see the paper for details.

5. Number of Instances: 768

6. Number of Attributes: 8 plus class 

7. For Each Attribute: (all numeric-valued)
   1. Number of times pregnant
   2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   3. Diastolic blood pressure (mm Hg)
   4. Triceps skin fold thickness (mm)
   5. 2-Hour serum insulin (mu U/ml)
   6. Body mass index (weight in kg/(height in m)^2)
   7. Diabetes pedigree function
   8. Age (years)
   9. Class variable (0 or 1)

8. Missing Attribute Values: Yes

9. Class Distribution: (class value 1 is interpreted as "tested positive for
   diabetes")

   Class Value  Number of instances
   0            500
   1            268
   
## Structure

### make dataset imbalance

making dataset imbalance will able you to evaluate your machine learning methods and data engineering methods under harder conditions. Although, you can skip this stage by modifying config.py .

### Data cleaning 

#### Missing value checking:

another stage that you can do it to see which samples is missing.

#### Detect outlier data: 

outlier detection by Median and Median Absolute Deviation Method (MAD). replacing the outlier by mean/median/most frequent values of that variable

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
