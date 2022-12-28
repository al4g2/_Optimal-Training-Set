"""
Scores prediction performance based on the Regional Scoring array
    Regional scoring is created from the predictions in 23_Predictions.dream3d


Binomial Blur array was created in dream3d
A binary regional array was created in dream3d by thresholding (Otsu) the Binomial Blur array
True values are scored vs the binary regional array

"""


import h5py
import pandas as pd
import numpy as np
from collections import Counter
from MetricsReport import metrics_report


##################################################
# ENTER PATHS and FILENAMES HERE!

# TRAINING
trained_part = 'Part-2'
train_size = '1'  # percentage of part section used for training
# train_size = '99'
trained_section = 'Cylinder'
model_type = 'LogRegCV'  # model type used for training
features = '4'  # feature model that was used for training

# TESTING
test_part = 'Part-1'
# test_part = 'Part-2'
# test_section = 'Cylinder'
# test_section = 'NomA'
test_section = 'Center'
# test_section = 'NomB'

# array names; from DREAM.3D
true = 'CTB1'
pred = f'CTpred {test_part}_Cylinder_trained_on_21_{trained_part}_{train_size}pct_{trained_section}_{model_type}_{features}'
reg = f'REGpred {test_part}_Cylinder_trained_on_21_{trained_part}_{train_size}pct_{trained_section}_{model_type}_{features}'
msk = 'Extended Center Mask'

# filename
fname = f'23_{test_part} {test_section} Predictions'

##################################################


# read dream3d file as hdf
f = h5py.File(f'{fname}.dream3d', 'r')
# CTB1: True Defects
truedefects = f[f'DataContainers/Fusion/Fused Attribute Matrix/{true}']
# Predicted Defects array
preddefects = f[f'DataContainers/Fusion/Fused Attribute Matrix/{pred}']
# Regional Scoring array
regionaldefects = f[f'DataContainers/Fusion/Fused Attribute Matrix/{reg}']
# Mask array
mask = f[f'DataContainers/Fusion/Fused Attribute Matrix/{msk}']

# select all 3 dimensions for a 3D array
truedefects = np.array(truedefects[:, :, :, 0])
preddefects = np.array(preddefects[:, :, :, 0])
regionaldefects = np.array(regionaldefects[:, :, :, 0])
mask = np.array(mask[:, :, :, 0])


# create a dataframe of the arrays
df = pd.DataFrame({'true': truedefects.ravel(),
                   'pred': preddefects.ravel(),
                   'region': regionaldefects.ravel(),
                   'm_true': mask.ravel()})
# print(f'df.shape before drop: {df.shape} \n')


# apply mask by removing cells where mask values are zero
df.drop(df[df['m_true'] == 0].index, inplace=True)


# create 1d array of each column from the dataframe
true = df.loc[:, 'true'].values
pred = df.loc[:, 'pred'].values
region = df.loc[:, 'region'].values

print(f'test part: {test_part}_{test_section}')
print('model name: ', f'21_{trained_part}_{train_size}pct_{trained_section}_{model_type}_{features}.pkl')


print('True shape: %s' % Counter(true))
print('Predict shape: %s' % Counter(pred))
print('Region shape: %s' % Counter(region))


# metrics report
metrics_report(true, pred, 'True Score of Predictions')
metrics_report(true, region, 'Regional Score of Predictions')
