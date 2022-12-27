"""
Models IR data and Laser Power vs CT thresholds
includes CT class balancing with SMOTE
used to test binary datasets
Section 3 in SFF manuscript for Model Development
"""

import time
import h5py
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from FeatureList import feature_list
from MetricsReport import metrics_report
import pickle
from Classifiers import model_fitting


# opens all keys in the hdf5 (dream3d) file
def traverse_datasets(hdf_file):
    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield path, item
            elif isinstance(item, h5py.Group):  # test for group (go down)
                yield from h5py_dataset_iterator(item, path)
    for path, _ in h5py_dataset_iterator(hdf_file):
        yield path


##################################################
# ENTER PATHS and FILENAMES HERE!

# name of the test part
# test_part = 'Part-1'
test_part = 'Part-2'

# file name of the dream3d dataset
# f_name = '19b_Cylinder'
f_name = '19b_Cylinder Cropped'
# f_name = '19b_Center'

# which feature model to use
features = '40'

# file name of the output model
# model_type = 'LogReg'
model_type = 'LogRegCV'
# model_type = 'MLP'
# model_type = 'SVC'
# model_type = 'RandomForest'
# model_type = 'GradientBoosting'
# model_type = 'AdaBoost'

# export model?
# export_model = 'yes'
export_model = 'no'

##################################################

data_name = f'{test_part}_{f_name}_{model_type}_{features}'

tic = time.perf_counter()

# load dataset
d = pd.DataFrame()  # define DataFrame, d
# traverse data file and open all arrays in the 'Fused Attribute Matrix'
with h5py.File(f'{test_part}/{f_name}.dream3d', 'r') as f5Im:
    for i, dset in enumerate(traverse_datasets(f5Im)):
        if dset.startswith('/DataContainers/Fusion/Fused Attribute Matrix/'):
            d[dset[46:]] = np.ravel(f5Im[f'{dset}'])  # add each raveled array as column to DataFrame

# print the keys of the created dataframe
# print(f'DataFrame keys: {d.keys()}')

# remove rows where all values of TAT are zero
d.drop(d[d['TAT'] == 0].index, inplace=True)
# print(d)

print(f'\nTime after loading data: {time.perf_counter() - tic:0.4f} seconds\n')

# split dataset into target variable and feature columns
y = d.CTB1  # target
print('Target: ', y.name)
print('Feature model: ', features)
feature_cols = feature_list(features)
X = d[feature_cols]  # features
print('Inputs: ', feature_cols)

# apply normalization to all features
norm = MinMaxScaler()
X_norm = norm.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns=feature_cols)
X_norm.head()

# split  train and test sets
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=0.3, random_state=0)

# summarize initial class distribution
print('Original dataset shape: %s' % Counter(y))
print('Training dataset shape: %s' % Counter(y_trn))
print('Test dataset shape: % s' % Counter(y_tst))

# perform synthetic oversampling on minority because class sizes are significantly imbalanced
# training data is over sampled - no reason to over sample the test set
over = SMOTE(sampling_strategy=0.5)  # 2:1 sampling strategy
X_trn, y_trn = over.fit_resample(X_trn, y_trn)
# summarize the new class distribution
print('Over-sampled training dataset shape: %s' % Counter(y_trn))


# fit mode; return y_prediction values and classifier for export
y_prd, classifier = model_fitting(model_type, X_trn, X_tst, y_trn, feature_cols, data_name)

# print metrics
metrics_report(y_tst, y_prd, data_name)


# # loop across model types
# # for i in ['LogReg', 'LogRegCV', 'MLP', 'SVC', 'RandomForest', 'GradientBoosting', 'AdaBoost']:
# for i in ['LogRegCV', 'RandomForest', 'GradientBoosting', 'AdaBoost']:
#     data_name = f'{test_part}_{f_name}_{i}_{features}'
#     y_prd, classifier = model_fitting(i, X_trn, X_tst, y_trn, feature_cols, data_name)
#     metrics_report(y_tst, y_prd, data_name)
#     print(f'Time after training: {time.perf_counter() - tic:0.4f} seconds')


# **************Save model to file**************
if export_model == 'yes':
    pkl_filename = f'{test_part}/21_{data_name}.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(classifier, file)


# MODIFY MODEL SELECTION THRESHOLD on LogReg models
if model_type == 'LogReg' or 'LogRegCV':
    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score, roc_curve
    import matplotlib.pyplot as plt
    predtst = classifier.predict_proba(X_tst)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_tst, predtst)
    # Plot FPRvsTRP against the Threshold
    # input array for desired thresholds to run
    thresharray = [0.60, 0.70, 0.80, 0.85, 0.9]
    for slcthresh in thresharray:
        print('\n Selected threshold: %s' % slcthresh)
        dfplot = pd.DataFrame({'Threshold': thresholds, 'False Positive Rate': fpr, 'False Negative Rate': 1.-tpr})
        dfplot.plot(x='Threshold', y=['False Positive Rate', 'False Negative Rate'], figsize=(10, 6))
        plt.plot([slcthresh, slcthresh], [0, 1.0])  # plots a line from given points
        plt.xlim(0, 1)
        plt.title(f'FPRvsTRP Against  Threshold '
                  f'\n Modified threshold = {slcthresh} '
                  f'\n Feature Model: {features} '
                  f'\n Target: {y.name}', wrap=True)
        plt.show()
        # New performance report due to modified threshold
        y_pred_thresh = np.where(predtst >= slcthresh, 1, 0)
        # new confusion matrix
        cm = confusion_matrix(y_tst, y_pred_thresh)
        print('New Confusion matrix (modified threshold): \n %s' % cm)
        # new classification report
        print('Classification Report (modified threshold): \n %s' % classification_report(y_tst, y_pred_thresh))
        # f1
        print('f1-score (modified threshold): ', f1_score(y_tst, y_pred_thresh))
        # ROC and FPRvsTRP
        roc_auc = roc_auc_score(y_tst, y_pred_thresh)
        print('ROC AUC (modified threshold): ', roc_auc)
        predtst = classifier.predict_proba(X_tst)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_tst, predtst)


toc = time.perf_counter()
print(f'\nElapsed time: {toc - tic:0.4f} seconds')
