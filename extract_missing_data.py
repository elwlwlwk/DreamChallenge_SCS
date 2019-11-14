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
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR

DATA_PATH = '../../data/complete_cell_lines'

complete_files = os.listdir(DATA_PATH)

train_files = complete_files
test_files = complete_files[-5:]

all_markers = ['b.CATENIN',
       'cleavedCas', 'CyclinB', 'GAPDH', 'IdU', 'Ki.67', 'p.4EBP1',
       'p.Akt.Ser473.', 'p.AKT.Thr308.', 'p.AMPK', 'p.BTK', 'p.CREB', 'p.ERK',
       'p.FAK', 'p.GSK3b', 'p.H3', 'p.JNK', 'p.MAP2K3', 'p.MAPKAPK2',
       'p.MEK', 'p.MKK3.MKK6', 'p.MKK4', 'p.NFkB', 'p.p38', 'p.p53',
       'p.p90RSK', 'p.PDPK1', 'p.RB', 'p.S6', 'p.S6K', 'p.SMAD23',
       'p.SRC', 'p.STAT1', 'p.STAT3', 'p.STAT5']
missing_marker = 'p.PLCg2'

treatments = ['EGF', 'iEGFR', 'iMEK', 'iPI3K', 'iPKC']
timestamps = [0.0, 5.5, 7.0, 9.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 23.0, 25.0, 30.0, 40.0, 60.0]


def regularize(df):
    min = df.min()
    return df[df != min].median().fillna(0)


def extract_full_data(data):
    full_data = data[data.treatment == 'full']
    full_median = regularize(full_data.sample(200))[all_markers]
    return full_median


def extract_whole_timed_data(data, target_treatment):
    data = data[data.treatment == target_treatment]
    existing_times = list(sorted(set(data.time)))
    missing_times = sorted(set(timestamps).difference(existing_times))

    timed_data = []
    whole_time_data = []
    for marker in all_markers:
        time_data = data.loc[:, ('time', marker)]
        min = time_data[marker].min()
        X, Y = zip(*time_data[time_data[marker] != min].sample(2000).values)
        X = list(map(lambda x: [x], X))
        clf = SVR(gamma='scale', C=1.0, epsilon=0.2)
        predict = clf.fit(X, Y).predict(list(map(lambda x: [x], timestamps)))
        whole_time_data.append(predict)
    return pd.DataFrame(list(zip(*whole_time_data)), columns = all_markers)



def extract_missing_marker_data(data, whole_time_data, target_marker, target_treatment):
    x = []
    y = []
    treatment_datum = list(map(lambda x: 1 if x == target_treatment else 0, treatments))
    existing_markers = list(filter(lambda x: x != target_marker, all_markers))
    for idx in range(len(timestamps)):
        rnn_datum = np.zeros((len(timestamps) - 1, len(all_markers)))
        time_datum = whole_time_data[0:idx].values
        if idx != 0:
            rnn_datum[-(idx):] = time_datum
        full_datum = extract_full_data(data)[existing_markers].values
        dnn_datum = whole_time_data.iloc[idx][existing_markers].values
        if target_marker != '':
            answer = whole_time_data.iloc[idx][target_marker]
        else:
            answer = 0


        x.append([rnn_datum, dnn_datum, full_datum, treatment_datum])
        y.append(answer)
    return [x, y]


def pool_wraper(args):
    return extract_missing_data(args[0], args[1], args[2])


def extract_missing_data(data, target_treatment, target_marker):
    # print(f'target file: {target_file}')
    # data = pd.read_csv(f'{DATA_PATH}/{target_file}')

    whole_time_data = extract_whole_timed_data(data, target_treatment)
    extracted_data = extract_missing_marker_data(data, whole_time_data, target_marker, target_treatment)
    return extracted_data


def main():
    data = list(map(lambda x: pd.read_csv(f'{DATA_PATH}/{x}'), train_files))
    print('data loaded')
    for _ in range(100):
        with Pool(16) as p:
            result = p.map(pool_wraper, product(data, treatments, [missing_marker]))
        with open(f'E:/missing_predict/{missing_marker}_median/train_data/{int(random() * 1000000)}.pkl','wb') as f:
            pickle.dump(result, f)
    pass

def test():
    data = list(map(lambda x: pd.read_csv(f'{DATA_PATH}/{x}'), train_files[0:1]))
    extract_missing_data(data[0], treatments[0], [])

if __name__ == '__main__':
    # main()
    test()