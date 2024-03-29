import configparser
import numpy as np
import os
import pandas as pd
import time

from data_loader import DataLoader
from feature_selector import BaselineSelector, AnalysisSelector
from utils import balanced_split, parse_config_features, plot_confusion_matrix, get_n_participants

def calculate_features(config: configparser.ConfigParser):

    feature_selector = BaselineSelector()
    all_features = parse_config_features(config)
    n_features = len(all_features)
    feature_selector.selectFeatures(**all_features)
    dataset_path = config['Data']['dataset_path']

    n_participants = get_n_participants(dataset_path)
    data = DataLoader(dataset_path, np.arange(n_participants))

    feature_names = []
    for f in all_features['features']:
        for i in range(19):
            feature_names.append(f + "_" + str(i))

    n_epochs = 0
    n_channels = 0
    features = None
    labels = None
    par_ids = None
    start_time = time.time()
    for curr in data:
        n_epochs += 1
        data = curr['data']
        par_id = curr['par_id']

        if not(n_channels): n_channels = data.shape[0]
        label = curr['group']     
        curr_features = feature_selector.transform(data)
        features = np.concatenate((features, curr_features)) if features is not None else curr_features

        labels = np.concatenate((labels, [label])) if labels is not None else [label]
        par_ids = np.concatenate((par_ids, [par_id])) if par_ids is not None else [par_id]

        if n_epochs % 1000 == 0:
            print("Time for {} epochs is {}".format(n_epochs, time.time() - start_time))

    features = features.reshape(n_epochs, -1)

    df = pd.DataFrame(features, columns=feature_names)
    
    if not os.path.exists("data/features"):
        os.makedirs("data/features")
    df.to_csv("data/features/features_new.csv")
    np.save("data/features/labels_new.npy", labels)
    np.save("data/features/par_ids_new.npy", par_ids)

if __name__ == '__main__':
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    calculate_features(config=config)
