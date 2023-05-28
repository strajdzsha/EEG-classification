import configparser
import numpy as np
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from data_loader import DataLoader
from feature_selector import BaselineSelector, AnalysisSelector
from utils import balanced_split, parse_config_features


def get_model(config: configparser.ConfigParser):
    """
    Returns model by name
    """
    hyperparams = config['Model']
    name = hyperparams['model_name']
    random_state = None if hyperparams['random_state'] == 'None' else int(hyperparams['random_state'])

    if name == 'logistic_regression':
        return LogisticRegression(random_state=random_state)
    
    elif name == 'knn':
        n_neighbors = int(hyperparams['n_neighbors'])
        return KNeighborsClassifier(n_neighbors=n_neighbors, random_state=random_state)
    
    elif name == 'svm':
        return SVC(random_state=random_state)
    
    elif name == 'random_forest':
        n_estimators = int(hyperparams['n_estimators'])
        max_depth = int(hyperparams['max_depth'])
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    
    elif name == 'xgboost':
        n_estimators = int(hyperparams['n_estimators'])
        max_depth = int(hyperparams['max_depth'])
        return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    
    elif name == 'naive_bayes':
        return GaussianNB(**hyperparams)
    
    else:
        raise ValueError(f'Unknown model name: {name}')
    
def get_metrics(y_true, y_pred):
    """
    Returns metrics for classification
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def get_selector(config: configparser.ConfigParser):
    """
    Returns feature selector by name
    """
    name = config['Features']['selector_name']
    if name == 'baseline':
        return BaselineSelector()
    elif name == 'analysis':
        return AnalysisSelector()
    else:
        raise ValueError(f'Unknown selector name: {name}')

def train(config: configparser.ConfigParser):
    """
    Trains the model on the given features and returns the metrics
    """
    n_folds = int(config['Train']['n_folds'])
    num_test_part = int(config['Train']['num_test_part'])

    train_acc = 0.0
    train_f1 = 0.0
    train_precision = 0.0
    train_recall = 0.0

    feature_selector = get_selector(config)
    
    all_features = parse_config_features(config)
    feature_selector.selectFeatures(**all_features)
    dataset_path = config['Data']['dataset_path']

    model = get_model(config)

    for k in range(n_folds):
        train_ids, test_ids = balanced_split(dataset_path, num_test_part=num_test_part)
        train_data = DataLoader(dataset_path, train_ids)
        test_data = DataLoader(dataset_path, test_ids)

        # prepare features for training
        train_features = None
        train_labels = None
        for curr in train_data:
            data = curr['data']
            label = curr['group']            

            curr_features = feature_selector.transform(data)
            train_features = np.concatenate((train_features, curr_features)) if train_features is not None else curr_features

            train_labels = np.concatenate((train_labels, [label])) if train_labels is not None else [label]

        # train model
        train_features = train_features.reshape(len(train_labels), -1)
        model.fit(train_features, train_labels)

        # prepare features for testing
        test_features = None
        test_labels = None
        for curr in test_data:
            data = curr['data']
            label = curr['group']

            curr_features = feature_selector.transform(data)
            test_features = np.concatenate((test_features, curr_features)) if test_features is not None else curr_features

            test_labels = np.concatenate((test_labels, [label])) if test_labels is not None else [label]

        # test model
        test_features = test_features.reshape(len(test_labels), -1)
        test_pred = model.predict(test_features)

        # calculate metrics
        metrics = get_metrics(test_labels, test_pred)
        pass

if __name__ == "__main__":
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)