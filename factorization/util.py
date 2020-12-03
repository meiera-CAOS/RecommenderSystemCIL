import numpy as np
import re
import pandas as pd
import os
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from math import sqrt


# valid are the coordinates of the validation set
def prediction_set(val_pos, pred):
    predictions = val_pos.copy()
    for x in range(0, len(val_pos), 1):
        string_pos = val_pos[x]
        temp = map(int, re.findall(r'\d+', string_pos))
        cor = list(map(int, temp))
        predictions[x] = pred[cor[0] - 1][cor[1] - 1]
    return predictions


def evaluate(imputed_m, val, name=""):
    val_pos = val[:, 0]
    val_ratings = val[:, 1]
    pred = prediction_set(val_pos, imputed_m)
    rmse = sqrt(mean_squared_error(pred, val_ratings))
    print(name + " RMSE: %f" % rmse)
    return rmse


def split_data(data, ratio=0.9):
    data = shuffle(data)
    mask = np.random.rand(len(data)) < ratio
    train_data = data[mask]
    validation_data = data[~mask]

    return train_data, validation_data


def load_submission_format():
    # load submission data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/sampleSubmission.csv'))
    pred = np.array(df)

    # p_s:position_submission, r_s:rating_submission
    p_s = pred[:, 0]
    r_s = pred[:, 1]
    return p_s, r_s


def create_submission(matrix, submission_name):
    p_s, r_s = load_submission_format()

    # fetch data from matrix
    for x in range(0, len(p_s), 1):
        string_pos = p_s[x]
        temp = map(int, re.findall(r'\d+', string_pos))
        cor = list(map(int, temp))
        r_s[x] = matrix[cor[0] - 1][cor[1] - 1]
    print("FINISHED: Predictions read from matrix")

    res = np.concatenate((p_s.reshape(-1, 1), r_s.reshape(-1, 1)), axis=1)
    df_out = pd.DataFrame(res)
    df_out.to_csv(submission_name, index=False, header=["Id", "Prediction"])
    print("FINISHED: ", submission_name, " has been created")


def load_data_train(split=False):
    # load test data
    dt = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/data_train.csv'))
    train_data = np.array(dt)
    if split:
        train_data, val_data = split_data(train_data)
        return fill_matrix_from_data(train_data), val_data

    else:
        return fill_matrix_from_data(train_data)


def fill_matrix_from_data(data):
    positions = data[:, 0]
    ratings = data[:, 1]
    # create empty matrix with 10'000 users and 1'000 items
    size = (10000, 1000)
    m_incomplete = np.empty(size)
    m_incomplete[:] = np.nan

    # fill matrix from data
    for x in range(0, len(positions), 1):
        string_pos = positions[x]
        # cor: [x, y]
        temp = map(int, re.findall(r'\d+', string_pos))
        cor = list(map(int, temp))
        m_incomplete[cor[0] - 1][cor[1] - 1] = ratings[x]
    print("FINISHED: Data loaded into matrix")
    return m_incomplete


def matrix_to_csv(matrix, filename):
    matrix_out = pd.DataFrame(matrix)
    matrix_out.to_csv(filename, index=False, header=False)
    return


def csv_to_matrix(filename):
    test_in = pd.read_csv(filename, header=None)
    return np.array(test_in)


def test_matrix(small=True, n=200, m=20, inner_rank=4):
    """
    Returns an incomplete matrix, a corresponding complete matrix and a missing mask,
    those can be used to test how well the imputation algorithms perform.
    :param small: per default return 5 x 5 example hardcoded below.
    If small is False, the following parameters can be set to choose the dimensions
    of the randomly generated test matrix.
    :param n: dim of matrix
    :param m: other dim of matrix
    :param inner_rank: inner rank, duh
    :return:
    """
    if small:
        partial_m = np.array([[5., 4., 0., 0., 0.],
                              [4., 0., 4., 0., 0.],
                              [0., 5., 4., 0., 1.],
                              [0., 0., 0., 5., 4.],
                              [0., 3., 0., 0., 5.]])

        partial_m[partial_m == 0] = np.nan

        full_m = np.array([[5., 4., 3., 2., 2.],
                           [4., 5., 4., 1., 3.],
                           [5., 5., 4., 1., 1.],
                           [2., 1., 2., 5., 4.],
                           [2., 3., 1., 4., 5.]])

        missing_mask = np.array([[0, 0, 1, 1, 1],
                                 [0, 1, 0, 1, 1],
                                 [1, 0, 0, 1, 0],
                                 [1, 1, 1, 0, 0],
                                 [1, 0, 1, 1, 0]])

    else:
        full_m = np.dot(np.random.randn(n, inner_rank), np.random.randn(inner_rank, m))
        missing_mask = np.random.rand(*full_m.shape) < 0.1
        partial_m = full_m.copy()
        partial_m[missing_mask] = np.nan
    return partial_m, full_m, missing_mask
