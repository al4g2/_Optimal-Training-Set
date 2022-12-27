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
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from collections import Counter


##################################################
# ENTER PATHS and FILENAMES HERE!

# which part to use?
partnum = 'Part-1'
# partnum = 'Part-2'

# which section of the part
# section = 'NomA'
# section = 'Center'
section = 'NomB'

# array names; from DREAM.3D
true = 'CTB1'
pred = f'CTpred {partnum}_Cylinder Cropped_trained_on_21b_Part-2_19b_Cylinder Cropped_LogRegCV_40'
reg = 'Pred Regional'
bb = 'Pred Binomial Blur'
msk = 'Extended Center Mask'

# filename
fname = '23_Predictions'

##################################################


# Standard metrics report; prints Confusion Matrix and Classification Report
def metrics_report(y_true, y_pred, title):
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f'\nConfusion matrix, {title}: \n %s' % cm)

    # precision/ recall/ f1
    print('precision: ', precision_score(y_true, y_pred))
    print('recall: ', recall_score(y_true, y_pred))
    print('f1-score: ', f1_score(y_true, y_pred))

    # Compute classification report
    # print(f'\nClassification Report, {title}: \n %s' % classification_report(y_true, y_pred))


# read dream3d file as hdf
f = h5py.File(f'{partnum}/{fname} {section}.dream3d', 'r')
# CTB1: True Defects
truedefects = f[f'DataContainers/Fusion/Fused Attribute Matrix/{true}']
# Predicted Defects array
preddefects = f[f'DataContainers/Fusion/Fused Attribute Matrix/{pred}']
# Regional Scoring array
regionaldefects = f[f'DataContainers/Fusion/Fused Attribute Matrix/{reg}']
# Binomial Blur array
binomialblur = f[f'DataContainers/Fusion/Fused Attribute Matrix/{bb}']
# Mask array
mask = f[f'DataContainers/Fusion/Fused Attribute Matrix/{msk}']

# select all 3 dimensions for a 3D array
truedefects = np.array(truedefects[:, :, :, 0])
preddefects = np.array(preddefects[:, :, :, 0])
regionaldefects = np.array(regionaldefects[:, :, :, 0])
binomialblur = np.array(binomialblur[:, :, :, 0])
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

print(f'{partnum} {section}')


print('True shape: %s' % Counter(true))
print('Predict shape: %s' % Counter(pred))
print('Region shape: %s' % Counter(region))


# metrics report
metrics_report(true, pred, 'True Score of Predictions')
metrics_report(true, region, 'Regional Score of Predictions')
