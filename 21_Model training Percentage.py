"""
Trains model on % of voxels in the input array
    uses selected input file
    outputs trained model as .pkl
Logistic regression model with Cross Validation
    Models IR data and Laser Power vs CT thresholds
    includes CT class balancing with SMOTE
used to test binary datasets
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
# trained_part = 'Part-1'
trained_part = 'Part-2'

# name of starting section of part
trained_section = 'Cylinder'

# what percentage of the part to train on
train_size = .01

# which feature model to use
features = '4'

# file name of the output model
model_type = 'LogRegCV'

# export model?
# export_model = 'yes'
export_model = 'no'

##################################################

# file name of the dream3d dataset
f_name = f'19b_{trained_part} {trained_section}'

# convert train_size to string
str_train_size = str(int(train_size*100))

# data name for export and chart titles
data_name = f'{trained_part}_{str_train_size}pct_{trained_section}_{model_type}_{features}'

tic = time.perf_counter()

# load dataset
d = pd.DataFrame()  # define DataFrame, d
# traverse data file and open all arrays in the 'Fused Attribute Matrix'
with h5py.File(f'{f_name}.dream3d', 'r') as f5Im:
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
X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=(1-train_size), random_state=0)

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


# **************Save model to file**************
if export_model == 'yes':
    pkl_filename = f'21_{data_name}.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(classifier, file)


toc = time.perf_counter()
print(f'\nElapsed time: {toc - tic:0.4f} seconds')
