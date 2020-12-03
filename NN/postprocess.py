import re

import numpy as np
import pandas as pd

from NN.utils import clip_ratings


def clip_submission(filepath_old, filepath_new):
    # load submission data
    df = pd.read_csv(filepath_old)
    pred = np.array(df)

    # p_s:position_submission, r_s:rating_submission
    positions = pred[:, 0]
    ratings = pred[:, 1]

    # clip ratings
    clip_ratings(ratings)

    # write submission
    res = np.concatenate((positions.reshape(-1, 1), ratings.reshape(-1, 1)), axis=1)
    df_out = pd.DataFrame(res)
    df_out.to_csv(filepath_new, index=False, header=["Id", "Prediction"])


# was used to get the core information out of the out files from the leonhard random search job
def pretty_print(file_to_read, out_file):
    regex_list = [[], []]
    for line in open(file_to_read):
        for m in re.finditer('achieved mse of (?P<mse>0\.\d+).*(?P<params>runtime .*)', line):
            regex_list[0].append(m.group('mse'))
            regex_list[1].append(m.group('params'))

    regex_arr = np.array(regex_list)
    m = regex_arr[:, np.argsort(regex_arr[0])]
    m = np.array(m)
    m = np.array(m)
    u = np.vstack((m[0], m[1]))
    u = pd.DataFrame(u).transpose()
    u.to_csv(out_file)
