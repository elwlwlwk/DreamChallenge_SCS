from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Concatenate, TimeDistributed, Dropout, Input, concatenate, Flatten, BatchNormalization, ReLU, Add, Activation
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from functools import reduce
import pickle
import numpy as np
import os
import random
import json
import tensorflow as tf
import keras.backend as K

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

rnn_input = Input(shape=(15,35), name='rnn_input')
rnn_model = LSTM(35, input_shape=(15,35), bias_regularizer=l2(0.01), kernel_regularizer=l2(0.01), return_sequences=True)(rnn_input)
rnn_model = Flatten()(rnn_model)

full_input = Input(shape=(35,), name='full_input')
full_model = Dense(35, activation=gelu, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(full_input)

treatment_input = Input(shape=(5,), name='treatment_input')
treatment_model = Dense(5, activation='softmax', kernel_initializer='random_uniform')(treatment_input)

merged = concatenate([rnn_model, full_model, treatment_model])

short_cut = merged
for i in range(5):
    if(i==0):
        merged = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(merged)
        merged = BatchNormalization()(merged)
        merged = Activation(gelu)(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(merged)
        merged = BatchNormalization()(merged)
        merged = Activation(gelu)(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(merged)
        merged = BatchNormalization()(merged)
        merged = Activation(gelu)(merged)

        short_cut = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(short_cut)
        short_cut = BatchNormalization()(short_cut)

        merged = Add()([merged, short_cut])
        merged = Activation(gelu)(merged)

        short_cut = merged
    else:
        merged = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(
            merged)
        merged = BatchNormalization()(merged)
        merged = Activation(gelu)(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(
            merged)
        merged = BatchNormalization()(merged)
        merged = Activation(gelu)(merged)
        merged = Dropout(0.3)(merged)
        merged = Dense(50, kernel_initializer='random_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(
            merged)
        merged = BatchNormalization()(merged)
        merged = Activation(gelu)(merged)

        merged = Add()([merged, short_cut])
        merged = Activation(gelu)(merged)

        short_cut = merged

merged = Dropout(0.3)(merged)
merged = Dense(35, activation='linear')(merged)
merged_model = Model(inputs=[rnn_input, full_input, treatment_input], outputs=merged)
train_loss = []
val_loss = []
merged_model.compile(optimizer=Adam(0.001), loss='mean_squared_error')

learn_cnt = 0

version = 'all_mk17'

if f'ensemble_model_{version}.h5' in os.listdir():
    print('loading previos model and weights')
    merged_model = load_model(f'ensemble_model_{version}.h5', custom_objects={"gelu": gelu})
    merged_model.load_weights(f'ensemble_model_weights_{version}_test.h5')

K.set_value(merged_model.optimizer.lr, 0.00001)

if f'loss_{version}.json' in os.listdir():
    loss_hist = json.load(open(f'loss_{version}.json', 'rb'))
    train_loss = loss_hist['train_loss']
    val_loss = loss_hist['val_loss']

while True:
    try:
        train_file = random.sample(os.listdir(f'train_data_all'), 1)[0]
        data = pickle.load(open(f'train_data_all/{train_file}', 'rb'))
        print(f'#{learn_cnt} {version} train_file: {train_file}')
    except:
        continue

    complete = data
    X, _ = zip(*complete)
    flatten_X = reduce(lambda a, b: a + b, X)
    random.seed(3)
    random.shuffle(flatten_X)
    transformed_X = list(map(lambda x: list(x), zip(*flatten_X)))
    rnn_X, Y, full_X, treatment_X = transformed_X

    hist = merged_model.fit([rnn_X, full_X, treatment_X], np.array(Y), epochs=10, verbose=1, validation_split=0.2)
    train_loss += hist.history['loss']
    val_loss += hist.history['val_loss']
    print(f'loss: {hist.history["loss"][-1]}, val_loss: {hist.history["val_loss"][-1]}')
    learn_cnt += 1
    if learn_cnt % 10 == 9:
        print('save model')
        json.dump({'train_loss': train_loss, 'val_loss': val_loss}, open(f'loss_{version}.json', 'wt'))
        merged_model.save(f'ensemble_model_{version}.h5', include_optimizer=True)
        merged_model.save_weights(f'ensemble_model_weights_{version}.h5')