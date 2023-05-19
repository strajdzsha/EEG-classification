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
from utils import balanced_split


def get_model(config: configparser.ConfigParser):
    """
    Returns model by name
    """
    hyperparams = config['Model']
    name = hyperparams['model_name']

    if name == 'logistic_regression':
        return LogisticRegression(**hyperparams)
    elif name == 'knn':
        return KNeighborsClassifier(**hyperparams)
    elif name == 'svm':
        return SVC(**hyperparams)
    elif name == 'random_forest':
        return RandomForestClassifier(**hyperparams)
    elif name == 'xgboost':
        return xgb.XGBClassifier(**hyperparams)
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
        return BaselineSelector(**config['Features'])
    elif name == 'analysis':
        return AnalysisSelector(**config['Features'])
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
    feature_selector.selectFeatures(**config['Features'])
    dataset_path = config['Data']['dataset_path']

    model = get_model(config)

    for k in range(n_folds):
        train_ids, test_ids = balanced_split(dataset_path, num_test_part=num_test_part)
        train_data = DataLoader(dataset_path, train_ids)
        test_data = DataLoader(dataset_path, test_ids)

    


if __name__ == "__main__":
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
