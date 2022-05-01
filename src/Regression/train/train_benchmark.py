import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score






def benchmarkModels(data, label):

    # Using StratifiedKFold for cross validation to find best performing model
    kf = StratifiedKFold(n_splits=7, random_state=None)
    # Testing with 6 Models

    lr = LogisticRegression(solver='liblinear')  # as dataset is small
    svc = SVC()
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier()
    rf = RandomForestClassifier()
    xgb = XGBClassifier()

    lr_accuracy = []
    svc_accuracy = []
    knn_accuracy = []
    dt_accuracy = []
    rf_accuracy = []
    xgb_accuracy = []

    for train_idx, test_idx in kf.split(data, label):
        X_train, X_test = data.iloc[train_idx, :], data.iloc[test_idx, :]
        y_train, y_test = label[train_idx], label[test_idx]

        # Logistic Regression
        lr.fit(X_train, y_train)
        lr_prediction = lr.predict(X_test)
        lr_acc = accuracy_score(lr_prediction, y_test)
        lr_accuracy.append(lr_acc)

        # SVC
        svc.fit(X_train, y_train)
        svc_prediction = svc.predict(X_test)
        svc_acc = accuracy_score(svc_prediction, y_test)
        svc_accuracy.append(svc_acc)

        # KNN
        knn.fit(X_train, y_train)
        knn_prediction = knn.predict(X_test)
        knn_acc = accuracy_score(knn_prediction, y_test)
        knn_accuracy.append(knn_acc)

        # Decision Tree
        dt.fit(X_train, y_train)
        dt_prediction = dt.predict(X_test)
        dt_acc = accuracy_score(dt_prediction, y_test)
        dt_accuracy.append(dt_acc)

        # Random Forest
        rf.fit(X_train, y_train)
        rf_prediction = rf.predict(X_test)
        rf_acc = accuracy_score(rf_prediction, y_test)
        rf_accuracy.append(rf_acc)

        # XGB Classifier
        xgb.fit(X_train, y_train)
        xgb_prediction = xgb.predict(X_test)
        xgb_acc = accuracy_score(xgb_prediction, y_test)
        xgb_accuracy.append(xgb_acc)

    print('Logistic Regression- Accuracy of each fold:', *lr_accuracy)
    print('Average accuracy of Logistic Regression: ', np.mean(lr_accuracy))
    print('Standard deviation of accuracy:', np.std(lr_accuracy))
    print('=' * 50)
    print('SVC- Accuracy of each fold:', *svc_accuracy)
    print('Average accuracy of SVC: ', np.mean(svc_accuracy))
    print('Standard deviation of accuracy:', np.std(svc_accuracy))
    print('=' * 50)
    print('KNN- Accuracy of each fold:', *knn_accuracy)
    print('Average accuracy of KNN: ', np.mean(knn_accuracy))
    print('Standard deviation of accuracy:', np.std(knn_accuracy))
    print('=' * 50)
    print('Decision Tree- Accuracy of each fold:', *dt_accuracy)
    print('Average accuracy of Decision Tree: ', np.mean(dt_accuracy))
    print('Standard deviation of accuracy:', np.std(dt_accuracy))
    print('=' * 50)
    print('Random Forest- Accuracy of each fold:', *rf_accuracy)
    print('Average accuracy of Random Forest: ', np.mean(rf_accuracy))
    print('Standard deviation of accuracy:', np.std(rf_accuracy))
    print('=' * 50)
    print('XGB Classifier- Accuracy of each fold:', *xgb_accuracy)
    print('Average accuracy of XGB Classifier: ', np.mean(xgb_accuracy))
    print('Standard deviation of accuracy:', np.std(xgb_accuracy))