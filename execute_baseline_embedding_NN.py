import argparse
import os
import sys

import numpy as np
import pandas as pd

from NN.preprocess import preprocess_input
from NN.baseline_embedding_NN import train, predict
from NN.utils import split_data

OUTPUT = os.path.join(os.path.dirname(__file__),'./baseline_embedding_NN.csv')

TRAIN_DATA = os.path.join(os.path.dirname(__file__),'./data/data_train.csv')
SUBMISSION_DATA = os.path.join(os.path.dirname(__file__),'./data/sampleSubmission.csv')
H5DATA = os.path.join(os.path.dirname(__file__),'./data/data.h5')

parser = argparse.ArgumentParser()
parser.add_argument('--no-train', action='store_true')

args = parser.parse_args()

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
    preprocess_input(TRAIN_DATA, SUBMISSION_DATA, H5DATA)

Data = pd.read_hdf(H5DATA, key='dataset')

if vars(args)['no_train']:
    model_name = os.path.join(os.path.dirname(__file__),'./NN/model/baseline_embedding_nn.hdf5')
else:
    model_name = os.path.join(os.path.dirname(__file__),'./NN/model/baseline_embedding_new_model.hdf5')
    train_set, test_set = split_data(Data)
    train(train_set, test_set, model_file=model_name)

PredictionTargets = pd.read_hdf(os.path.join(os.path.dirname(__file__),'./data/data.h5'), key='to_predict')

predictions = predict(PredictionTargets, model_name)

# P_s:position_submission
P_s = PredictionTargets['Id']

# vertically stack position and predictions and write them out to disk
res = np.vstack((P_s, predictions))
df_out = pd.DataFrame(res).transpose()
df_out.to_csv(OUTPUT, index=False, header=["Id", "Prediction"])
print("FINISHED: " + OUTPUT + " has been created")
