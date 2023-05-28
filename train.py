import configparser
import numpy as np
import time
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from data_loader import DataLoader
from feature_selector import BaselineSelector, AnalysisSelector
from utils import balanced_split, parse_config_features, shape_wrapper


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

    feature_selector = get_selector(config)
    
    all_features = parse_config_features(config)
    feature_selector.selectFeatures(**all_features)
    dataset_path = config['Data']['dataset_path']

    model = get_model(config)
    absolute_start_time = time.time()
    start_time = time.time()
    bs = 32

    for k in range(n_folds):
        print(f'Fold {k}/{n_folds}')
        print('Started preparing train features')
        train_ids, test_ids = balanced_split(dataset_path, num_test_part=num_test_part)
        train_data = DataLoader(dataset_path, train_ids, batch_size=bs)
        test_data = DataLoader(dataset_path, test_ids, batch_size=bs)

        n_features = feature_selector.transform(train_data[0]['data']).shape[0] # ugly hack
        # prepare features for training
        train_features = np.zeros((len(train_data), n_features))
        train_labels = None
        for i, curr_batch in enumerate(train_data):
            if not isinstance(curr_batch, list): # for batch_size = 1 case
                curr_batch = [curr_batch]

            data = [d['data'] for d in curr_batch]
            data = np.stack(data) # (batch_size, num_channels, num_samples)

            label = [d['group'] for d in curr_batch]

            sample_shape = (data.shape[1], data.shape[2])
            data = data.reshape(data.shape[0], -1)

            # efficient way to apply transform to each row in batch
            curr_features = np.apply_along_axis( # (batch_size, num_features)
                shape_wrapper,
                axis=1,
                arr=data,
                shape = sample_shape,
                func = feature_selector.transform)
            

            train_features[i*bs:(i+1)*bs] = curr_features

            train_labels = label if train_labels is None else train_labels + label

        end_time = time.time()
        print(f'Finished preparing train features: {end_time - start_time} seconds')

        train_features = train_features[:len(train_labels)]

        # train model
        start_time = time.time()
        print('Started training model')
        
        model.fit(train_features, train_labels)

        end_time = time.time()
        print(f'Finished training model: {end_time - start_time} seconds')
        
        train_acc = 0.0
        train_f1 = 0.0
        train_precision = 0.0
        train_recall = 0.0

        
        print('Begin testing')
        start_time = time.time()

        # prepare features for testing
        test_features = np.zeros((len(test_data), n_features))
        test_labels = None
        
        for i, curr_batch in enumerate(test_data):
            if not isinstance(curr_batch, list):
                curr_batch = [curr_batch]

            data = [d['data'] for d in curr_batch]
            data = np.stack(data) # (batch_size, num_channels, num_samples)

            label = [d['group'] for d in curr_batch]

            sample_shape = (data.shape[1], data.shape[2])
            data = data.reshape(data.shape[0], -1)

            # efficient way to apply transform to each row in batch
            curr_features = np.apply_along_axis( # (batch_size, num_features)
                shape_wrapper,
                axis=1,
                arr=data,
                shape = sample_shape,
                func = feature_selector.transform)
            
            test_features[i*bs: (i+1)*bs] = curr_features

            test_labels = label if test_labels is None else test_labels + label

        test_features = test_features[:len(test_labels)]

        # test model
        test_pred = model.predict(test_features)

        # calculate metrics
        metrics = get_metrics(test_labels, test_pred)
        
        train_acc += metrics['accuracy']
        train_f1 += metrics['f1']
        train_precision += metrics['precision']
        train_recall += metrics['recall']

        end_time = time.time()
        print(f'Finished testing: {end_time - start_time} seconds')
        
        print('Current fold metrics:')
        print(f'Accuracy: {metrics["accuracy"]}')
        print(f'Precision: {metrics["precision"]}')
        print(f'Recall: {metrics["recall"]}')
        print(f'F1: {metrics["f1"]}')

    train_acc /= n_folds
    train_f1 /= n_folds
    train_precision /= n_folds
    train_recall /= n_folds

    print('--' * 20)
    print('Average metrics:')
    print(f'Accuracy: {train_acc}')
    print(f'Precision: {train_precision}')
    print(f'Recall: {train_recall}')
    print(f'F1: {train_f1}')

    absolute_end_time = time.time()
    print(f'Total training time: {absolute_end_time - absolute_start_time} seconds')

if __name__ == "__main__":
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    train(config)