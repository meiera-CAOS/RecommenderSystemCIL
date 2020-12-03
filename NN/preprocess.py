import re

import numpy as np
import pandas as pd

from NN.utils import k_approx

"""
    Reads data in csv files and writes DataFrames to data.h5 in working directory. This file is read then by the NN.
    IN: data_train, sample_submission file paths
    OUT: None, writes to @Input{output_file}
"""


def input_mf_data(mf_data, dataset, to_predict):
    """
    Input the data from the matrix factorization into the data frame, so that it can be used in predictions
    """
    # add mf prediction at position (i,u) to prediction data
    mf = [mf_data[i, u] for (i, u) in zip(to_predict.item_id, to_predict.user_id)]
    to_predict['mf_pred'] = mf
    # dimension reduction to k
    k = 50
    R = k_approx(np.matrix(mf_data), k)
    # add mf prediction at position (i,u) to train data (with reduced dimensionality)
    dataset['mf_pred'] = [R[i, u] for (i, u) in zip(dataset.item_id, dataset.user_id)]


def preprocess_input(data_train, sample_submission, output_file):
    dt = pd.read_csv(data_train)
    ds = pd.read_csv(sample_submission)

    # creates two columns, one for the user and one for the movie indices
    split_user_movie_index(dt)
    split_user_movie_index(ds)

    # assigns user ids between 0 and 999 (not 1 .. 1000) same for items
    create_indices(dt)
    create_indices(ds)

    # no prediction
    del ds['Prediction']

    dt.to_hdf(output_file, key='dataset')
    ds.to_hdf(output_file, key='to_predict')


def create_indices(dt):
    dt.user_id = dt.user_id.astype('category').cat.codes.values
    dt.item_id = dt.item_id.astype('category').cat.codes.values


def split_user_movie_index(ds):
    us_id, vs_id = [], []
    for index in ds['Id'].values:
        tmp = list(map(int, re.findall(r'\d+', index)))
        us_id.append(tmp[1])
        vs_id.append(tmp[0])
    ds['user_id'] = us_id
    ds['item_id'] = vs_id
