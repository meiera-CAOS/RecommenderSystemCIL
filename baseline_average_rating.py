import os
import re
import sys

import numpy as np
import pandas as pd

NUM_USERS = 10000
NUM_ITEMS = 1000

TRAIN_DATA = './data/data_train.csv'
SUBMISSION_DATA = './data/sampleSubmission.csv'

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


def csv_to_indices_and_ratings(file_name):
    dataFrame = pd.read_csv(file_name)
    dataArray = np.array(dataFrame)

    PositionStrings = dataArray[:, 0]
    Ratings = dataArray[:, 1]

    return PositionStrings, Ratings


def position_string_to_list_of_indices(position_string):
    # parse string_pos into two integers, row and column position
    temp = map(int, re.findall(r'\d+', position_string))
    return list(map(int, temp))


# load test data
(PositionStrings, Ratings) = csv_to_indices_and_ratings(
    os.path.join(os.path.dirname(__file__), TRAIN_DATA))

# create empty matrix with 10'000 users and 1'000 items
size = (NUM_USERS, NUM_ITEMS)
DataMatrix = np.zeros(size)

# fill matrix with known data
for x in range(len(PositionStrings)):
    position = position_string_to_list_of_indices(PositionStrings[x])
    # subtract 1 from both position indices because matrix is 0-based, strings are 1-based
    DataMatrix[position[0] - 1][position[1] - 1] = Ratings[x]
print("FINISHED: Known data loaded into matrix")

'''
##################################################################################
fill unknown entries with average values
'''
# Calculate average rating per movie. Use nan to represent unknown ratings and ignore those for average calculation
KnownData = DataMatrix.copy()
KnownData[KnownData == 0] = np.nan
Averages = np.nanmean(KnownData[:, :], axis=0)

# Fill data matrix with averages
for x in range(NUM_USERS):
    for y in range(NUM_ITEMS):
        if DataMatrix[x][y] == 0:
            DataMatrix[x][y] = Averages[y]
print("FINISHED: Zero entries filled")

'''
##################################################################################
create submission file from predictions
'''
(PositionsSubmission, RatingsSubmission) = csv_to_indices_and_ratings(
    os.path.join(os.path.dirname(__file__), SUBMISSION_DATA))

# fetch data from matrix
for x in range(len(PositionsSubmission)):
    position = position_string_to_list_of_indices(PositionsSubmission[x])
    RatingsSubmission[x] = DataMatrix[position[0] - 1][position[1] - 1]
print("FINISHED: Predictions read from matrix")

res = np.concatenate((PositionsSubmission.reshape(-1, 1), RatingsSubmission.reshape(-1, 1)), axis=1)
print(res)
df_out = pd.DataFrame(res)
df_out.to_csv('baseline_average_ratings.csv', index=False, header=["Id", "Prediction"])
print("FINISHED: average_ratings.csv has been created")
