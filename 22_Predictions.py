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
        if s == 'NomA':
            return 0
        elif s == 'Center':
            return 276
        elif s == 'NomB':
            return 694
        elif s == 'Cylinder Cropped':
            return 10
    if p == 'Part-2':
        if s == 'NomA':
            return 0
        elif s == 'Center':
            return 286
        elif s == 'NomB':
            return 706
        elif s == 'Cylinder Cropped':
            return 10


# function to export array as hdf file
def hdf_array(data_array, output_shape, z_c, folder, name):
    hdf_out = np.zeros(output_shape)  # initialize return array
    # print('hdf_out.shape: ', hdf_out.shape)
    a2 = np.column_stack((data_array.X, data_array.Y, data_array.Z, data_array.pred))  # 2D array
    # print(f'a2:\n{a2}')
    # subtract 1 from x, y, z value as the array is 0-base
    for row in a2:
        row[0] = row[0] - 1
        row[1] = row[1] - 1
        row[2] = row[2] - 1
    # convert values in predict_results in integer
    a2 = a2.astype(int)
    # build 3d matrix, hdf_out
    for row in a2:
        i = row[0]
        j = row[1]
        k = row[2] - z_c
        hdf_out[k, j, i] = row[3]  # 3D array
    # create HDF file output
    hf = h5py.File(f'{folder}/22_{name}.h5', 'w')
    hf.create_dataset(f'CTpred {name}', data=hdf_out)


##################################################
# ENTER PATHS and FILENAMES HERE!

# name of the section the model was trained on
trained_part = 'Part-2'
# trained_section = 'Center'
trained_section = 'Cylinder Cropped'

# name of the section to be tested
test_part = 'Part-1'
# test_part = 'Part-2'
# test_section = 'NomA'
# test_section = 'Center'
# test_section = 'Cylinder Cropped'
test_section = 'NomB'

# which feature model to use
features = '40'

# model type that was used for training
# model_type = 'LogReg'
model_type = 'LogRegCV'
# model_type = 'MLP'
# model_type = 'GradientBoosting'
# model_type = 'RandomForest'

# LogReg threshold
slcthresh = [0.85]

##################################################


data_name = f'{test_part}_{test_section}_trained_on_21b_{trained_part}_19b_{trained_section}_{model_type}_{features}'

# starting layer of each test part/section combo
z_crop = crop_layer(test_part, test_section)

# Load trained model from file
model_filename = f'21b_{trained_part}_19b_{trained_section}_{model_type}_{features}.pkl'
with open(model_filename, 'rb') as file:
    model = pickle.load(file)
print('model name: ', model_filename)


# load test dataset
d = pd.DataFrame()  # define DataFrame, d
# traverse data file and open all arrays in the 'Fused Attribute Matrix'
with h5py.File(f'{test_part}/19b_{test_section}.dream3d', 'r') as f5Im:
    # add each raveled array as column to DataFrame
    for w, dset in enumerate(traverse_datasets(f5Im)):
        if dset.startswith('/DataContainers/Fusion/Fused Attribute Matrix/'):
            # print(f5Im[f'{dset}'].shape)
            d[dset[46:]] = np.ravel(f5Im[f'{dset}'])
    # shape of test dataset (used to export HDF)
    hdf_shape = np.array(f5Im['/DataContainers/Fusion/Fused Attribute Matrix/CTB1'][:, :, :, 0]).shape
    # print('hdf_shape: ', hdf_shape)

# rename the "CTB1 Regional Scoring" key
d.rename(columns={'CTB1 Regional Scoring': 'Regional'}, inplace=True)

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


# PREDICTION PROBABILITY
predp = model.predict_proba(X)[:, 1] * 100  # probability of one
# output hdf with predict results
# hdf_array(predp, 'Probability')


# BINARY PREDICTIONS at different thresholds (0.5 is the standard)
if model_type == 'LogReg' or 'LogRegCV':
    pred = np.where(model.predict_proba(X)[:, 1] >= slcthresh, 1, 0)
else:
    pred = model.predict_proba(X)[:, 1]  # NEED TO ADD FOR OTHER MODELS

# add predictions as column in DataFrame
d['pred'] = pred
# print('d.shape: ', d.shape)

# metrics report for predictions
print('\n\nTrue Score of Predictions: \n')
metrics_report(y, pred, data_name)

# metrics report for Regions
print('\n\nRegional Score of Predictions: \n')
metrics_report(d.Regional, pred, data_name)

# output hdf with predict results
# hdf_array(d, hdf_shape, z_crop, test_part, data_name)
