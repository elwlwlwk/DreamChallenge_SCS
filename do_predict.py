import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
from multiprocessing import Pool
from itertools import product
import pickle
from random import random
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Concatenate, TimeDistributed, Dropout, Input, concatenate, Flatten, BatchNormalization, ReLU
from keras.regularizers import l1, l2
from keras.optimizers import Adam
from functools import reduce
import pickle
import numpy as np
import os
import random
import json
import tensorflow as tf

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

from extract_missing_data import regularize

all_markers = ['b.CATENIN',
       'cleavedCas', 'CyclinB', 'GAPDH', 'IdU', 'Ki.67', 'p.4EBP1',
       'p.Akt.Ser473.', 'p.AKT.Thr308.', 'p.AMPK', 'p.BTK', 'p.CREB', 'p.ERK',
       'p.FAK', 'p.GSK3b', 'p.H3', 'p.JNK', 'p.MAP2K3', 'p.MAPKAPK2',
       'p.MEK', 'p.MKK3.MKK6', 'p.MKK4', 'p.NFkB', 'p.p38', 'p.p53',
       'p.p90RSK', 'p.PDPK1', 'p.RB', 'p.S6', 'p.S6K', 'p.SMAD23',
       'p.SRC', 'p.STAT1', 'p.STAT3', 'p.STAT5']
treatments = ['EGF', 'iEGFR', 'iMEK', 'iPI3K', 'iPKC']
timestamps = [0.0, 5.5, 7.0, 9.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 23.0, 25.0, 30.0, 40.0, 60.0]

DATA_PATH = 'ch4'

ch4_files = os.listdir(DATA_PATH)

model = load_model(f'ensemble_model_all_mk17.h5', custom_objects={"gelu": gelu})
model.load_weights(f'ensemble_model_weights_all_mk17_1.h5')

SAMPLES = 100

def extract_full_data(data):
    X = []
    treatments = [[1 if i == k else 0 for i in range(5)] for k in range(5)]
    for treatment in treatments:
        for i in range(SAMPLES):
            rnn_data = np.zeros((len(timestamps) - 1, len(all_markers)))
            full_data = regularize(data.sample(500))[all_markers]
            X.append([rnn_data, full_data.values, treatment])

    return X

def predict(flatten_X):
    result = []
    transformed_X = list(map(lambda x: list(x), zip(*flatten_X)))
    rnn_X, full_X, treatment_X = transformed_X
    for idx in range(1, len(timestamps)):
        predicted = model.predict([rnn_X, full_X, treatment_X])
        for row_idx in range(len(rnn_X)):
            rnn_X[row_idx][-idx] = predicted[row_idx]
    last_predict = model.predict([rnn_X, full_X, treatment_X])
    for idx in range(len(rnn_X)):
        rnn_X[idx] = np.append(rnn_X[idx], [last_predict[idx]], axis=0)
    for idx in range(len(treatments)):
        result.append(np.array(rnn_X[idx * SAMPLES : (idx + 1) * SAMPLES]).mean(axis=0))
    return result

if __name__ == '__main__':
    ch4_data = list(map(lambda x: pd.read_csv(f'{DATA_PATH}/{x}'), ch4_files))
    result = pd.DataFrame(columns = ['cell_line', 'treatment', 'time'] + all_markers)
    for data in ch4_data:
        full_data = extract_full_data(data)
        predict_result = predict(full_data)

        cell_line_result = []
        for treatment_idx in range(len(treatments)):
            for time_idx in range(len(timestamps)):
                predicted = predict_result[treatment_idx][time_idx]
                for idx in range(len(predicted)):
                    min_val = data[all_markers[idx]].min()
                    min_cnt = list(data[all_markers[idx]].values).count(min_val)
                    min_ratio = min_cnt / len(data[all_markers[idx]])
                    predicted[idx] = predicted[idx] * (1 - min_ratio) + min_val * min_ratio
                cell_line_result.append([data.iloc[0]['cell_line'], treatments[treatment_idx], timestamps[time_idx]] + list(predicted))
        cell_line_pd = pd.DataFrame(cell_line_result, columns = ['cell_line', 'treatment', 'time'] + all_markers)
        result = result.append(cell_line_pd)

    template = pd.read_csv('subchallenge_4_template_data.csv')
    filtered_result = pd.DataFrame(columns = ['cell_line', 'treatment', 'time'] + all_markers)
    for idx, template_row in template.iterrows():
        filtered_result = filtered_result.append(result[(result.treatment == template_row.treatment) & (result.cell_line == template_row.cell_line) & (result.time == template_row.time)])

    filtered_result.to_csv('ch4_result.csv', index=False)
    pass