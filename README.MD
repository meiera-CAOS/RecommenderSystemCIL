# CIL-Project MeAndTheBoys
This is the codebase used for the practical project for the CIL course 2019 of the team MeAndTheBoys. 
It is a submission for the [Collaborative Filtering task](https://inclass.kaggle.com/c/cil-collab-filtering-2019)

## Required Libraries
* fancyimpute
* sklearn 
* Numpy
* scipy
* Tensorflow
* h5py (should be included in Tensorflow)
* Pandas

## Files in this repository
#### main.py
This file can be used to run the complete pipeline to use our framework.
When it is run without any arguments, it runs on the provided model,
which was used to generate the last submission, and outputs the 
submission data in a file called `submission_MeAndTheBoys.csv`

Parameters:
* --train: train a new neural network to use for the predictions
* -o <filename>: Specify the name of the produced sumbission file (the file extension '.csv' is added automatically)

#### execute_baseline_embedding_NN.py
This file can be used to generate the baseline submission using only
the unaltered neural network (using GMF). It trains a Neural Network
and then uses it to produce a submittable file (`baseline_embedding_NN.csv`)

Parameters:
* --no-train: use the provided model instead of training

#### execute_baseline_average_rating.py
This file can be used to generate a submission file that uses
the average ratings of the movies (`baseline_average_ratings.csv`)

#### main_MLF_MF_random_search.py
This file was used to find the optimal parameters for our NN

## Folders
### data
This folder is used to store data that is needed for predicting 
ratings. This includes the necessary inputs described below and also
files that are calculated during the process that will be reused
in a later step (results of the matrix imputation)

#### data_train.csv
This file contains the training data provided by the project.
#### sampleSubmission.csv
This file is a sample submission provided by the project. It is mainly used to parse
what users and items should be combined.

### factorization
Contains the files to produce a fully imputed data matrix
#### collaboration_filtering.py
Contains the code that imputes the complete Users/Items/Rating matrix.
It uses a combination of several Imputation methods to achieve this.
In the end it stores a file with the completed matrix in the data folder
(`linear_comb.csv`)
#### util.py
Contains some helper functions, that can read and write data from/to disk

### NN
This folder contains the models and code to train and run the 
Neural Network part of our framework
#### model
This folder contains the stored models. It contains the pretrained
models we produced to generate our baselines and final submission.
It is also used to store models that are created during training
##### baseline_embedding_nn.hdf5
The model used to generate the baseline that only uses the NN
##### pretrained_embeddingNN.hdf5
The model used to generate our final kaggle submission.

#### baseline_embedding_NN.py
This file contains the baseline NN logic.

#### embedding_noGMF_NN.py
This file contains the logic for the NN we used in our framework
to produce the submissions

#### postprocess.py
Contains utilities to change the output of our NN, such as clipping
the data, should it exceed the boundaries of [1,5]

#### preprocess.py
This file contains utilities that transform the given input files
such that they can be used in our framework

#### utils.py
Contains mostly functions to load and save data.