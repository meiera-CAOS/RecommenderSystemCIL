"""
   Implementation is based heavily on the code taken from: 
   https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html
"""

import warnings
import os

import numpy as np
import tensorflow.python.keras as keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

from NN.utils import load_model_from_file, split_data

warnings.filterwarnings('ignore')

EMBEDDING_NN_MODEL_FILE = os.path.join(os.path.dirname(__file__),'./model/pretrained_embedding.hdf5')
PARAMS = dict()
PARAMS['epochs'] = 20
PARAMS['activation'] = 'relu'
PARAMS['dropout_rate'] = 0.1
PARAMS['init'] = 'normal'
PARAMS['optimizer'] = 'Adam'
PARAMS['n_latent_factors_user'] = 8
PARAMS['n_latent_factors_movie'] = 32
PARAMS['neurons_embed'] = [64, 64, 16, 16]
PARAMS['neurons_combine'] = [64, 32]
PARAMS['patience'] = 15


def predict(data, model_file=EMBEDDING_NN_MODEL_FILE):
    model = load_model_from_file(model_file)
    return model.predict([data.user_id, data.item_id, data.mf_pred]).flatten()


def train(train_set, test_set, params=PARAMS, model_file=EMBEDDING_NN_MODEL_FILE):
    n_users, n_movies = 1000, 10000
    DROPOUT = params['dropout_rate']

    # INPUT layers
    movie_input = keras.layers.Input(shape=[1], name='Item')
    user_input = keras.layers.Input(shape=[1], name='User')
    pred_mf_input = keras.layers.Input(shape=[1], name='MF_pred')

    # EMBEDDING layers for users and movies
    movie_embedding_mlp = keras.layers.Embedding(n_movies + 1, params['n_latent_factors_movie'],
                                                 name='Movie-Embedding-MLP')(movie_input)
    movie_vec_mlp = keras.layers.Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
    movie_vec_mlp = keras.layers.Dropout(DROPOUT)(movie_vec_mlp)

    user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(
        keras.layers.Embedding(n_users + 1, params['n_latent_factors_user'], name='User-Embedding-MLP')(user_input))
    user_vec_mlp = keras.layers.Dropout(DROPOUT)(user_vec_mlp)

    # MLP
    concat = keras.layers.merge.concatenate([movie_vec_mlp, user_vec_mlp], name='Concat')
    concat_dropout = keras.layers.Dropout(DROPOUT)(concat)
    dense = keras.layers.Dense(params['neurons_embed'][0],
                               kernel_initializer=params['init'], name='FullyConnected')(concat_dropout)
    dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
    dropout_1 = keras.layers.Dropout(DROPOUT, name='Dropout-1')(dense_batch)
    dense_2 = keras.layers.Dense(params['neurons_embed'][1],
                                 kernel_initializer=params['init'], name='FullyConnected-1')(dropout_1)
    dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)
    dropout_2 = keras.layers.Dropout(DROPOUT, name='Dropout-2')(dense_batch_2)
    dense_3 = keras.layers.Dense(params['neurons_embed'][2],
                                 kernel_initializer=params['init'], name='FullyConnected-2')(dropout_2)
    dense_4 = keras.layers.Dense(params['neurons_embed'][3],
                                 kernel_initializer=params['init'], name='FullyConnected-3', activation='relu')(dense_3)

    pred_mlp = keras.layers.Dense(1, activation=params['activation'],
                                  kernel_initializer=params['init'], name='Activation')(dense_4)

    combine_mlp_mf = keras.layers.merge.concatenate([pred_mf_input, pred_mlp], name='Concat-MF-MLP', axis=-1)
    result_combine = keras.layers.Dense(params['neurons_combine'][0],
                                        kernel_initializer=params['init'], name='Combine-MF-MLP')(combine_mlp_mf)
    deep_combine = keras.layers.Dense(params['neurons_combine'][1],
                                      kernel_initializer=params['init'], name='FullyConnected-4')(result_combine)

    result = keras.layers.Dense(1, kernel_initializer=params['init'], name='Prediction')(deep_combine)

    model = keras.Model([user_input, movie_input, pred_mf_input], result)
    # opt = keras.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=params['optimizer'], loss='mean_squared_error', metrics=['accuracy'])

    # The patience parameter is the amount of epochs to wait for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

    # saves the model and the weights at the specified model_file path
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint, early_stop]

    model.fit([train_set.user_id, train_set.item_id, train_set.mf_pred], train_set.Prediction,
              validation_data=([test_set.user_id, test_set.item_id, test_set.mf_pred], test_set.Prediction),
              epochs=params['epochs'], verbose=1, callbacks=callbacks_list, shuffle=True)


def optimize(dataset, model_file):
    ########################################################
    # Parameters to try
    params = dict()
    params['epochs'] = [20]
    params['patience'] = [8]
    params['activation'] = ['relu']  # 'tanh', 'sigmoid'
    params['learn_rate'] = [0.001]  # 0.01, 0.1, 0.2
    params['dropout_rate'] = [0.1]
    params['init'] = ['uniform', 'glorot_uniform']
    params['optimizer'] = ['SGD', 'Adam']
    params['n_latent_factors_user'] = [8, 16, 32]
    params['n_latent_factors_movie'] = [8, 16, 32]

    # pick a random init of params
    for key, value in params.items():
        params[key] = (np.random.choice(value))

    # neurons set at random - such that the #neurons of layer is never more than the one before

    params['neurons_embed'] = [np.random.choice([64, 128])]  # sets max #neurons of MLP embedding
    # set remaining 3 #neurons of dense layer
    for i in range(1, 4):
        max_val = params['neurons_embed'][i - 1]
        if i < 3:
            options = [16, 32, 64, 128]
        else:
            options = [8, 16, 32, 64, 128]
        options = [i for i in options if i <= max_val]
        choice = np.random.choice(options)
        params['neurons_embed'].append(choice)

    # set #neurons of MF-MLP combinations
    options = [32, 64]
    params['neurons_combine'] = [np.random.choice(options)]  # sets max #neurons of MF-MLP combination
    # set 2nd #neurons for MF-MLP combination
    max_val = params['neurons_combine'][0]
    options.append(16)
    options = [i for i in options if i <= max_val]
    choice = np.random.choice(options)
    params['neurons_combine'].append(choice)

    ##############################################################
    #  model file default - not needed in this function,
    train_set, test_set = split_data(dataset)
    print("Starting training for params:", params)
    train(train_set, test_set, params, model_file)

    return train_set, test_set, params
