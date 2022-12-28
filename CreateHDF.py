import numpy as np
import h5py


# function to export array as hdf file
def output_hdf(data_array, output_shape, z_c, name):
    hdf_out = np.zeros(output_shape)  # initialize return array
    # print('hdf_out.shape: ', hdf_out.shape)
    a2 = np.column_stack((data_array.X, data_array.Y, data_array.Z, data_array.pred))  # 2D array
    # print(f'a2:\n{a2}')
    # print('a2.shape: ', a2.shape)
    # subtract 1 from x, y, z value as the array is 0-base
    for row in a2:
        row[0] = row[0] - 1
        row[1] = row[1] - 1
        row[2] = row[2] - 1
    # convert values in predict_results in integer
    # print(f'a2:\n{a2}')
    a2 = a2.astype(int)
    # build 3d matrix, hdf_out
    for row in a2:
        # print(row)
        i = row[0]
        j = row[1]
        k = row[2] - z_c
        hdf_out[k, j, i] = row[3]  # 3D array
    # create HDF file output
    hf = h5py.File(f'22_{name}.h5', 'w')
    hf.create_dataset(f'CTpred {name}', data=hdf_out)


# function to export array as hdf file
def output_hdf_old(data_array, output_shape, z_c, folder, name):
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
