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

from keras.models import load_model
import tensorflow as tf

from extract_missing_data import extract_missing_data
from predict_missing import get_predicted_missing


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

all_markers = ['b.CATENIN',
       'cleavedCas', 'CyclinB', 'GAPDH', 'IdU', 'Ki.67', 'p.4EBP1',
       'p.Akt.Ser473.', 'p.AKT.Thr308.', 'p.AMPK', 'p.BTK', 'p.CREB', 'p.ERK',
       'p.FAK', 'p.GSK3b', 'p.H3', 'p.JNK', 'p.MAP2K3', 'p.MAPKAPK2',
       'p.MEK', 'p.MKK3.MKK6', 'p.MKK4', 'p.NFkB', 'p.p38', 'p.p53',
       'p.p90RSK', 'p.PDPK1', 'p.RB', 'p.S6', 'p.S6K', 'p.SMAD23',
       'p.SRC', 'p.STAT1', 'p.STAT3', 'p.STAT5']
missing_marker = 'p.HER2'
treatments = ['EGF', 'iEGFR', 'iMEK', 'iPI3K', 'iPKC']

ALL_PATH = '../../data/all'
# COMPLETE_PATH = '../../data/complete_cell_lines'
# HER2_PATH = '../../data/p.HER2_incomplete'
# PLCG2_PATH = '../../data/p.PLCg2_incomplete'

all_files = os.listdir(ALL_PATH)
# complete_files = os.listdir(COMPLETE_PATH)
# her2_files = os.listdir(HER2_PATH)
# plcg2_files = os.listdir(PLCG2_PATH)

def predict_to_complete(predicted_data, missing_idx):
    X, Y = predicted_data
    for i in range(len(X)):
        X[i][1] = np.concatenate((X[i][1][0:missing_idx], [Y[i]], X[i][1][missing_idx:]))
        X[i][2] = np.concatenate((X[i][2][0:missing_idx], [Y[0]], X[i][2][missing_idx:]))
    return [X, Y]

def extract_missing_data_wraper(args):
    print('extract full: ', args[0].iloc[0]['cell_line'], args[1], args[2])
    return extract_missing_data(args[0], args[1], args[2])

def predict_missing_wraper(args):
    print('predicting: ', args[0].iloc[0]['cell_line'], args[1], args[2])
    model = load_model(f'ensemble_model_{args[2]}.h5', custom_objects={"gelu": gelu})
    model.load_weights(f'ensemble_model_weights_{args[2]}.h5')
    predicted = get_predicted_missing(args[0], args[1], args[2], model)
    return predict_to_complete(predicted, all_markers.index(args[2]))

if __name__ == '__main__':
    all_data = list(map(lambda x: pd.read_csv(f'{ALL_PATH}/{x}'), all_files))
    # complete_data = list(map(lambda x: pd.read_csv(f'{COMPLETE_PATH}/{x}'), complete_files))
    # print('complete loaded')
    # her2_data = list(map(lambda x: pd.read_csv(f'{HER2_PATH}/{x}'), her2_files))
    # print('her2 loaded')
    # plcg2_data = list(map(lambda x: pd.read_csv(f'{PLCG2_PATH}/{x}'), plcg2_files))
    # print('plcg2 loaded')


    for _ in range(200):
        with Pool(4) as p:
            extracted_all_data = p.map(extract_missing_data_wraper, product(all_data, treatments, ['']))
            # extracted_complete_data = p.map(extract_missing_data_wraper, product(complete_data, treatments, ['']))
            # her2_complete_data = p.map(predict_missing_wraper, product(her2_data, treatments, ['p.HER2']))
            # plcg2_complete_data = p.map(predict_missing_wraper, product(plcg2_data, treatments, ['p.PLCg2']))

        with open(f'E:/ch4/train_data_all/{int(random() * 1000000)}.pkl', 'wb') as f:
            # pickle.dump([extracted_complete_data, her2_complete_data, plcg2_complete_data], f)
            pickle.dump(extracted_all_data, f)

    print('done')