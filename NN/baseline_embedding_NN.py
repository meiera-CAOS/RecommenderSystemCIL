"""
   Implementation is based heavily on the code taken from: 
   https://nipunbatra.github.io/blog/2017/neural-collaborative-filtering.html
"""

import os

import tensorflow.python.keras as keras
from tensorflow.python.keras.callbacks import ModelCheckpoint

from NN.utils import load_model_from_file

EMBEDDING_NN_MODEL_FILE = os.path.join(os.path.dirname(__file__), './model/baseline_embedding_nn.hdf5')
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
    return model.predict([data.user_id, data.item_id]).flatten()


def train(train_set, test_set, params=PARAMS, model_file=EMBEDDING_NN_MODEL_FILE):
    n_latent_factors_mf = 50
    n_users, n_movies = 1000, 10000
    DROPOUT = params['dropout_rate']

    movie_input = keras.layers.Input(shape=[1], name='Item')
    movie_embedding_mlp = keras.layers.Embedding(n_movies + 1, params['n_latent_factors_movie'],
                                                 name='Movie-Embedding-MLP')(movie_input)
    movie_vec_mlp = keras.layers.Flatten(name='FlattenMovies-MLP')(movie_embedding_mlp)
    movie_vec_mlp = keras.layers.Dropout(DROPOUT)(movie_vec_mlp)

    movie_embedding_mf = keras.layers.Embedding(n_movies + 1, n_latent_factors_mf, name='Movie-Embedding-MF')(
        movie_input)
    movie_vec_mf = keras.layers.Flatten(name='FlattenMovies-MF')(movie_embedding_mf)
    movie_vec_mf = keras.layers.Dropout(DROPOUT)(movie_vec_mf)

    user_input = keras.layers.Input(shape=[1], name='User')
    user_vec_mlp = keras.layers.Flatten(name='FlattenUsers-MLP')(
        keras.layers.Embedding(n_users + 1, params['n_latent_factors_user'], name='User-Embedding-MLP')(user_input))
    user_vec_mlp = keras.layers.Dropout(DROPOUT)(user_vec_mlp)

    user_vec_mf = keras.layers.Flatten(name='FlattenUsers-MF')(
        keras.layers.Embedding(n_users + 1, n_latent_factors_mf, name='User-Embedding-MF')(user_input))
    user_vec_mf = keras.layers.Dropout(DROPOUT)(user_vec_mf)

    concat = keras.layers.merge.concatenate([movie_vec_mlp, user_vec_mlp], name='Concat')
    concat_dropout = keras.layers.Dropout(DROPOUT)(concat)
    dense = keras.layers.Dense(params['neurons_embed'][0], name='FullyConnected',
                               kernel_initializer=params['init'])(concat_dropout)
    dense_batch = keras.layers.BatchNormalization(name='Batch')(dense)
    dropout_1 = keras.layers.Dropout(DROPOUT, name='Dropout-1')(dense_batch)
    dense_2 = keras.layers.Dense(params['neurons_embed'][1], name='FullyConnected-1',
                                 kernel_initializer=params['init'])(dropout_1)
    dense_batch_2 = keras.layers.BatchNormalization(name='Batch-2')(dense_2)

    dropout_2 = keras.layers.Dropout(DROPOUT, name='Dropout-2')(dense_batch_2)
    dense_3 = keras.layers.Dense(params['neurons_embed'][2], name='FullyConnected-2',
                                 kernel_initializer=params['init'])(dropout_2)
    dense_4 = keras.layers.Dense(params['neurons_embed'][3], name='FullyConnected-3', activation=params['activation'],
                                 kernel_initializer=params['init'])(dense_3)

    pred_mf = keras.layers.merge.dot([movie_vec_mf, user_vec_mf], name='Dot', axes=-1)

    pred_mlp = keras.layers.Dense(1, activation=params['activation'], name='Activation',
                                  kernel_initializer=params['init'])(dense_4)

    combine_mlp_mf = keras.layers.merge.concatenate([pred_mf, pred_mlp], name='Concat-MF-MLP', axis=-1)
    result_combine = keras.layers.Dense(params['neurons_combine'][0], name='Combine-MF-MLP',
                                        kernel_initializer=params['init'])(combine_mlp_mf)
    deep_combine = keras.layers.Dense(params['neurons_combine'][1], name='FullyConnected-4',
                                      kernel_initializer=params['init'])(result_combine)

    result = keras.layers.Dense(1, name='Prediction', kernel_initializer=params['init'])(deep_combine)

    model = keras.Model([user_input, movie_input], result)
    model.compile(optimizer=params['optimizer'], loss='mean_squared_error', metrics=['accuracy'])

    # The patience parameter is the amount of epochs to wait for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=params['patience'])
    # saves the model and the weights at the specified model_file path
    checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    callbacks_list = [checkpoint, early_stop]  # saves current best model & early stop

    model.fit([train_set.user_id, train_set.item_id], train_set.Prediction,
              validation_data=([test_set.user_id, test_set.item_id], test_set.Prediction),
              epochs=params['epochs'], verbose=0, callbacks=callbacks_list, shuffle=True)
