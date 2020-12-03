import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model


def cosine_sim(M, outer=False):
    if outer:
        sim_mat = np.dot(M, M.T)
    else:
        sim_mat = np.dot(M.T, M)
    # normalize matrix and return
    return (sim_mat - np.min(sim_mat)) / (np.max(sim_mat) - np.min(sim_mat))


def split_data(dataset, test_size=0.1):
    train, test = train_test_split(dataset, test_size=test_size)
    return train, test


def load_model_from_file(model_filename):
    loaded_model = load_model(model_filename)
    print("FINISHED: Loading NN model from disk")
    return loaded_model


def k_approx(M, k):
    u, s, vh = np.linalg.svd(M, full_matrices=True)

    uk = u[:, :k]
    sk = s[:k]
    vhk = vh[:k, :]

    return np.dot(uk, np.dot(np.diag(sk), vhk))


def save_predictions_to_file(filename, to_predict, predictions):
    # P_s:position_submission
    P_s = to_predict['Id']

    # vertically stack position and predictions and write them out to disk
    res = np.vstack((P_s, predictions))
    df_out = pd.DataFrame(res).transpose()
    df_out.to_csv(filename, index=False, header=["Id", "Prediction"])


def clip_ratings(data):
    data[data > 5] = 5
    data[data < 1] = 1
    return data

