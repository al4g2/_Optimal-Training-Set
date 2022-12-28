"""
Trains model on sections of the input array
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import MinMaxScaler
from FeatureList import feature_list
import pickle


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

# name of the trained part
# trained_part = 'Part-1'
trained_part = 'Part-2'

# name of starting section of part
trained_section = 'Cylinder'

# what layers of the part to train on
#   whole cylinder is 0:970
#   M section is 286:342
#   S&T section is 582:705
#   T section is 649:705
#   O section is 435:491
l1 = 286
l2 = 705

# which feature model to use
features = '4'

# file name of the output model
model_type = 'LogRegCV'

# export model?
export_model = 'yes'
# export_model = 'no'

##################################################

# file name of the dream3d dataset
f_name = f'19b_{trained_part} {trained_section}'

# data name for export and chart titles
data_name = f'{trained_part}_100pct_{trained_section}_{l1}_to_{l2}_{model_type}_{features}'

tic = time.perf_counter()

# load dataset
d = pd.DataFrame()  # define DataFrame, d
# traverse data file and open all arrays in the 'Fused Attribute Matrix'
with h5py.File(f'{f_name}.dream3d', 'r') as f5Im:
    for i, dset in enumerate(traverse_datasets(f5Im)):
        if dset.startswith('/DataContainers/Fusion/Fused Attribute Matrix/'):
            d[dset[46:]] = np.ravel(f5Im[f'{dset}'])  # add each raveled array as column to DataFrame

# keep only rows where Z is >= l1 and <= l2
d = d[(d['Z'] >= l1) & (d['Z'] <= l2)]

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
# print('Inputs: ', feature_cols)

# apply normalization to all features
norm = MinMaxScaler()
X_norm = norm.fit_transform(X)
X_norm = pd.DataFrame(X_norm, columns=feature_cols)
X_norm.head()

# split  train and test sets
X_trn = X
y_trn = y

# summarize initial class distribution
print('Original dataset shape: %s' % Counter(y))
print('Training dataset shape: %s' % Counter(y_trn))

# perform synthetic oversampling on minority because class sizes are significantly imbalanced
# training data is over sampled - no reason to over sample the test set
over = SMOTE(sampling_strategy=0.5)  # 2:1 sampling strategy
X_trn, y_trn = over.fit_resample(X_trn, y_trn)
# summarize the new class distribution
print('Over-sampled training dataset shape: %s' % Counter(y_trn))


# fit mode; classifier for export
classifier = LogisticRegressionCV(cv=20, scoring='f1', random_state=0, n_jobs=-1, max_iter=2000000000)
classifier.fit(X_trn, y_trn)  # fit the model with the data


# **************Save model to file**************
if export_model == 'yes':
    pkl_filename = f'21_{data_name}.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(classifier, file)


toc = time.perf_counter()
print(f'\nElapsed time: {toc - tic:0.4f} seconds')
