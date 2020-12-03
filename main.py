import argparse
import os
import sys

import numpy as np
import pandas as pd

from NN import preprocess, embedding_noGMF_NN
## ----------- CONSTANTS -----------
from NN.utils import split_data, clip_ratings, save_predictions_to_file
from factorization.collaboration_filtering import impute

TRAIN_DATA = os.path.join(os.path.dirname(__file__), './data/data_train.csv')
SUBMISSION_DATA = os.path.join(os.path.dirname(__file__), './data/sampleSubmission.csv')
H5DATA = os.path.join(os.path.dirname(__file__), './data/data.h5')
OUTPUT_FILE = 'submission_MeAndTheBoys'

MF_PREDICTION_PATH = os.path.join(os.path.dirname(__file__), './data/matrix_factorization.csv')
MF_COMPLETE_DATA_PATH = os.path.join(os.path.dirname(__file__), './data/linear_comb.csv')

PRE_TRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), './NN/model/pretrained_embeddingNN.hdf5')
SELF_TRAINED_MODEL_PATH = os.path.join(os.path.dirname(__file__), './NN/model/selftrained_embeddingNN.hdf5')

## ----------- MAIN CODE -----------
# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help="if this parameter is present, train a new network")
parser.add_argument('-o', action='store',
                    help="the filename without file ending (.csv will be appended) that should be given to the output "
                         "file (default: %(default)s).", default=OUTPUT_FILE)

args = parser.parse_args()
args_dict = vars(args)
DO_TRAIN = args_dict['train']
if args_dict['o']:
    OUTPUT_FILE = args_dict['o']

COMPLETE_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), OUTPUT_FILE + ".csv")
# Check if the given filename is valid
try:
    with open(COMPLETE_OUTPUT_PATH, 'w+') as temp:
        pass
        # should throw error if file cannot be created
    # Remove the file once again to have clear output
    os.remove(COMPLETE_OUTPUT_PATH)
except OSError:
    sys.exit("The given output file name '" + OUTPUT_FILE + "' is not valid. Exiting now!")

# Check if inputs are present and parse them into a h5 file
if not os.path.isdir(os.path.join(os.path.dirname(__file__),'./data')):
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

## Process available data into an imputed matrix
print("Starting to impute matrix")
impute(MF_COMPLETE_DATA_PATH)
print("Matrix Imputation over!")

print("Loading dataset and to_predict set from h5 file and add matrix data to it")
matrix_data = np.matrix(pd.read_csv(MF_COMPLETE_DATA_PATH, header=None))
data = pd.read_hdf(H5DATA, key='dataset')
indices_to_predict = pd.read_hdf(H5DATA, key='to_predict')
preprocess.input_mf_data(matrix_data, data, indices_to_predict)
print("Finished adding information to data container file")

## NN part
# Train the model if desired
model_path = SELF_TRAINED_MODEL_PATH if DO_TRAIN else PRE_TRAINED_MODEL_PATH

if DO_TRAIN:
    train_data, test_data = split_data(data)
    embedding_noGMF_NN.train(train_data, test_data, model_file=model_path)

if not os.path.isfile(model_path):
    sys.exit(model_path + " is not available in 'NN/model' folder! Exiting now!")

predictions = embedding_noGMF_NN.predict(indices_to_predict, model_file=model_path)
clip_ratings(predictions)
save_predictions_to_file(COMPLETE_OUTPUT_PATH, indices_to_predict, predictions)
print("DONE")
