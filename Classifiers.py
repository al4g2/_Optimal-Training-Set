"""
Logistic regression classifier
uses cross validation with n=20 k-folds
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
"""


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import pandas as pd
import matplotlib.pyplot as plt


# Logistic Regression
def logreg(X_train, X_test, y_train, feature_cols, data_name):
    classifier = LogisticRegression(max_iter=2000000000)  # create the instance of the model
    classifier.fit(X_train, y_train)  # fit the model with the data
    y_pred = classifier.predict(X_test)  # Predicting test set results
    # print('y_pred shape: %s' % Counter(y_pred))

    # feature importance
    coefficients = classifier.coef_[0]
    print('coefficients', coefficients)
    # summarize coefficients
    sort_cos = pd.DataFrame()
    for i, v in enumerate(coefficients):
        add = pd.DataFrame([feature_cols[i], v])
        add = add.T
        sort_cos = sort_cos.append(add)
    sort_cos.columns = ['Features', 'Coefficients']
    s_coefficients = sort_cos.sort_values('Coefficients', ascending=False)
    # print(s_coefficients)
    ax = s_coefficients.plot.bar()
    plt.xticks([x for x in range(len(coefficients))], s_coefficients.iloc[:, 0])
    plt.title(f'Feature Coefficients (normalized)\n{data_name}')
    plt.show()

    return y_pred, classifier


# Logistic Regression, with cross validation
def logreg_cv(X_train, X_test, y_train, feature_cols, data_name):
    classifier = LogisticRegressionCV(cv=20, scoring='f1', random_state=0, n_jobs=-1, max_iter=2000000000)
    classifier.fit(X_train, y_train)  # fit the model with the data
    y_pred = classifier.predict(X_test)  # Predicting test set results
    # print('y_pred shape: %s' % Counter(y_pred))

    # # feature importance
    # coefficients = classifier.coef_[0]
    # print('coefficients', coefficients)
    # # summarize coefficients
    # sort_cos = pd.DataFrame()
    # for i, v in enumerate(coefficients):
    #     add = pd.DataFrame([feature_cols[i], v])
    #     add = add.T
    #     sort_cos = sort_cos.append(add)
    # sort_cos.columns = ['Features', 'Coefficients']
    # s_coefficients = sort_cos.sort_values('Coefficients', ascending=False)
    # # print(s_coefficients)
    # ax = s_coefficients.plot.bar()
    # plt.xticks([x for x in range(len(coefficients))], s_coefficients.iloc[:, 0])
    # plt.title(f'Feature Coefficients (normalized)\n{data_name}')
    # plt.show()

    return y_pred, classifier


# other classifiers; not logistic regression
def other_classifiers(X_train, X_test, y_train, model_type):
    if model_type == 'MLP':
        classifier = MLPClassifier(activation='logistic')
    elif model_type == 'SVC':
        classifier = LinearSVC(penalty='l1', dual=False, max_iter=100000)
    elif model_type == 'RandomForest':
        classifier = RandomForestClassifier(n_jobs=-1)
    elif model_type == 'GradientBoosting':
        classifier = GradientBoostingClassifier()
    elif model_type == 'AdaBoost':
        classifier = AdaBoostClassifier()
    else:
        print(f'INCORRECT model_type\nRUNNING LOGISTIC REGRESSION')
        classifier = LogisticRegression()

    classifier.fit(X_train, y_train)  # fit the model with the data
    y_pred = classifier.predict(X_test)  # Predicting test set results

    return y_pred, classifier


def model_fitting(model_type, X_train, X_test, y_train, feature_cols, data_name):
    if model_type == 'LogReg':
        y_p, clssfr = logreg(X_train, X_test, y_train, feature_cols, data_name)
    elif model_type == 'LogRegCV':
        y_p, clssfr = logreg_cv(X_train, X_test, y_train, feature_cols, data_name)
    else:
        y_p, clssfr = other_classifiers(X_train, X_test, y_train, model_type)

    return y_p, clssfr
