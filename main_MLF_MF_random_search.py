import sys
import time
import os

import numpy as np
import pandas as pd

from NN import preprocess
from NN.embedding_noGMF_NN import predict, optimize
from NN.utils import load_model_from_file, k_approx, save_predictions_to_file, clip_ratings

TRAIN_DATA = './data/data_train.csv'
SUBMISSION_DATA = './data/sampleSubmission.csv'
H5DATA = './data/data.h5'

# Check if inputs are present and parse them into a h5 file
if not os.path.isdir('./data'):
    sys.exit("No 'data' folder found, exiting now!")
missing_files = False
if not os.path.isfile(TRAIN_DATA):
    print("Did not find 'train_data.csv' in folder 'data'!")
    missing_files = True
if not os.path.isfile(SUBMISSION_DATA):
    print("Did not find 'sampleSubmission.csv' in folder 'data'!")
    missing_files = True
if missing_files:
    sys.exit("There are missing files! Exiting now!")

if not os.path.isfile(H5DATA):
    print("'data.h5' is not present in folder 'data'. Building it now")
    preprocess.preprocess_input(TRAIN_DATA, SUBMISSION_DATA, H5DATA)
# read data, read MF
data = pd.read_hdf(os.path.join(os.path.dirname(__file__), './data/data.h5'), key='dataset')
to_predict = pd.read_hdf(os.path.join(os.path.dirname(__file__), './data/data.h5'), key='to_predict')
print("FINISHED: Data loaded into matrix")

M = pd.read_csv(os.path.join(os.path.dirname(__file__), './data/linear_comb.csv'), header=None)
M = np.matrix(M)

# add mf prediction at position (i,u) to prediction data
mf = [M[i, u] for (i, u) in zip(to_predict.item_id, to_predict.user_id)]
to_predict['mf_pred'] = mf
# dimension reduction to k
k = 50
R = k_approx(np.matrix(M), k)
# add mf prediction at position (i,u) to train data (with reduced dimensionality)
data['mf_pred'] = [R[i, u] for (i, u) in zip(data.item_id, data.user_id)]

# train NW loop (with random parameters)
mse = 5
best_model = ''
best_params = dict()
for i in range(10):
    start_time = time.time()
    model_file = os.path.join(os.path.dirname(__file__), './NN/model/embeddingNN_noGMF_model_' + str(i) + '.best.hdf5')
    train_set, test_set, params = optimize(data, model_file)
    # we want the best model (including weights) of the past training iteration
    model = load_model_from_file(model_file)
    tmp_mse, _ = model.evaluate([data.user_id, data.item_id, data.mf_pred], data.Prediction)  # measure mse
    end_time = time.time()
    print("achieved mse of %s, runtime %s min, params %s" % (tmp_mse, (end_time - start_time) / 60, params))
    # save current best model
    if tmp_mse < mse:
        best_model = model_file
        best_params = params
        mse = tmp_mse
        with open(os.path.join(os.path.dirname(__file__), './NN/parameters.txt'), 'a+') as file:
            file.write('MSE: ' + str(mse) + ', params: ' + str(params) + '\n')

# predict best model
predictions = predict(data=to_predict, model_file=best_model)
print("predicting data for best model with parameters: , achieving mse: ")
# create submission for best model.
clip_ratings(predictions)
save_predictions_to_file(os.path.join(os.path.dirname(__file__), './NN/submission_MeAndTheBoys_RS.csv'), to_predict,
                         predictions)
print("Done")
sys.exit(0)
