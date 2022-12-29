"""
Applies a pre-made trained model to another dataset
--> works on the Cylinder model: starting at Z-0 and ending at Z-1011

creates an array of the prediction labels for visualization
  outputs HDF5 file with predictions

Used for Section 4 in SFF manuscript for Predictive Model
"""

import h5py
import pickle
import pandas as pd
import numpy as np
from FeatureList import feature_list
from MetricsReport import metrics_report
from CreateHDF import output_hdf


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


# returns z_crop layer based on the cropping of each part/section combo
def crop_layer(p, s):
    if p == 'Part-1':
        if s == 'Cylinder':
            return 10
        elif s == 'NomA':
            return 10
        elif s == 'Center':
            return 286
        elif s == 'NomB':
            return 704
        elif s == 'Cylinder Cropped':
            return 10
    if p == 'Part-2':
        if s == 'Cylinder':
            return 10
        elif s == 'NomA':
            return 10
        elif s == 'Center':
            return 296
        elif s == 'NomB':
            return 716
        elif s == 'Cylinder Cropped':
            return 10


##################################################
# ENTER PATHS and FILENAMES HERE!

# TRAINING
trained_part = 'Part-2'
train_size = '100'  # percentage of part section used for training
trained_section = 'Cylinder'
l1 = 286  # layer range for models trained with a specific section *note* this is just the range for trained model
l2 = 705  # disregard if train_size <> 100
model_type = 'LogRegCV'  # model type used for training
features = '4'  # feature model that was used for training

# TESTING
test_part = 'Part-1'
# test_part = 'Part-2'
test_section = 'Cylinder'
# test_section = 'NomA'
# test_section = 'Center'
# test_section = 'NomB'

# LogReg threshold
slcthresh = [0.85]

##################################################

# print the name of the test part and test section
print(f'test part: {test_part}_{test_section}')

# starting layer of each test part/section combo
z_crop = crop_layer(test_part, test_section)

# Load trained model from file
if train_size == '100':
    model_filename = f'21_{trained_part}_{train_size}pct_{trained_section}_{l1}_to_{l2}_{model_type}_{features}'
else:
    model_filename = f'21_{trained_part}_{train_size}pct_{trained_section}_{model_type}_{features}'
with open(f'{model_filename}.pkl', 'rb') as file:
    model = pickle.load(file)
print('model name: ', model_filename)

# data name for export and chart titles
data_name = f'{test_part}_{test_section}_trained_on_{model_filename}'


# load test dataset
d = pd.DataFrame()  # define DataFrame, d
# traverse data file and open all arrays in the 'Fused Attribute Matrix'
with h5py.File(f'19b_{test_part} {test_section}.dream3d', 'r') as f5Im:
    # add each raveled array as column to DataFrame
    for w, dset in enumerate(traverse_datasets(f5Im)):
        if dset.startswith('/DataContainers/Fusion/Fused Attribute Matrix/'):
            # print(f5Im[f'{dset}'].shape)
            d[dset[46:]] = np.ravel(f5Im[f'{dset}'])
    # shape of test dataset (used to export HDF)
    hdf_shape = np.array(f5Im['/DataContainers/Fusion/Fused Attribute Matrix/CTB1'][:, :, :, 0]).shape
    print('test hdf_shape: ', hdf_shape)

# # rename the "CTB1 Regional Scoring" key
# d.rename(columns={'CTB1 Regional Scoring': 'Regional'}, inplace=True)

# print the keys of the created dataframe
# print(f'DataFrame keys: {d.keys()}')

# remove rows where all values of TAT are zero
d.drop(d[d['TAT'] == 0].index, inplace=True)
# print('d.shape: ', d.shape)

# split dataset into target variable and feature columns
y = d.CTB1  # target
print('Target: ', y.name)
print('Feature model: ', features)
feature_cols = feature_list(features)
X = d[feature_cols]  # features
# print('Inputs: ', feature_cols)


# BINARY PREDICTIONS at different thresholds (0.5 is the standard)
if model_type == 'LogReg' or 'LogRegCV':
    pred = np.where(model.predict_proba(X)[:, 1] >= slcthresh, 1, 0)
else:
    pred = model.predict_proba(X)[:, 1]  # NEED TO ADD FOR OTHER MODELS

# add predictions as column in DataFrame
d['pred'] = pred
# print('d.shape: ', d.shape)
# print(d['Z'])

# print the keys of the created dataframe
# print(f'DataFrame keys: {d.keys()}')

# metrics report for predictions
# print('\n\nTrue Score of Predictions: \n')
metrics_report(d['CTB1'], d['pred'], data_name)


# output hdf with predict results
output_hdf(d, hdf_shape, z_crop, data_name)
