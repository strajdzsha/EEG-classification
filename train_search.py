import configparser
import numpy as np
import time
import xgboost as xgb
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from scipy.stats import uniform
from pycm import ConfusionMatrix

from data_loader import DataLoader
from feature_selector import BaselineSelector, AnalysisSelector
from utils import balanced_split, parse_config_features, shape_wrapper, plot_confusion_matrix
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

def get_model(model_name):
    """
    Returns model by name
    """

    if model_name == 'logistic_regression':
        return LogisticRegression()
    
    elif model_name == 'knn':
        return KNeighborsClassifier()
    
    elif model_name == 'svm':
        return SVC()
    
    elif model_name == 'random_forest':
        return RandomForestClassifier()
    
    elif model_name == 'xgboost':
        return xgb.XGBClassifier()
    
    elif model_name == 'naive_bayes':
        return GaussianNB()
    
    else:
        raise ValueError(f'Unknown model model: {model_name}')
    
def get_metrics(y_true, y_pred):
    """
    Returns metrics for classification
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'confusion_matrix': ConfusionMatrix(y_true, y_pred)
    }


if __name__ == "__main__":
    config_path = './config.ini'
    config = configparser.ConfigParser()
    config.read(config_path)
    n_folds = int(config['Train']['n_folds'])
    num_test_part = int(config['Train']['num_test_part'])
    print("n_folds is {} and num_test_part is {}".format(n_folds, num_test_part))

    X = pd.read_csv(config['Data']['features_path'])
    X.drop('Unnamed: 0', axis=1, inplace=True)

    with open('top_features.txt', 'r') as file:
        top_features = file.read().splitlines()
    X = X[top_features]
    print(X.shape)

    par_ids = np.load(config['Data']['par_ids_path'])
    X['par_ids'] = par_ids

    dataset_path = config['Data']['dataset_path']
    myCViterator = []
    for i in range(n_folds):
        train_ids, test_ids = balanced_split(dataset_path, num_test_part=num_test_part)
        train_idx = np.where(np.array(X['par_ids'].isin(train_ids)) == True)[0]
        test_idx = np.where(np.array(X['par_ids'].isin(test_ids)) == True)[0]
        myCViterator.append((train_idx, test_idx))
    X = X.drop('par_ids', axis=1)

    y = np.load(config['Data']['labels_path'])


    model_names = ['svm'] # podesiti za koje modele se radi
    for model_name in model_names:
        model = get_model(model_name)

        pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
        ])

        print("Starting grid search...")
        param_grid = {'kernel': [''], 'model__learning_rate':[0.01, 0.05, 0.1]} # podesiti po zelji; prima i distribucije 
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, average='macro'),
            'recall': make_scorer(recall_score, average='macro'),
            'f1': make_scorer(f1_score, average='macro')
        }
        clf = RandomizedSearchCV(pipeline, param_grid, cv = myCViterator, scoring=scoring, n_iter=3, refit='f1') # podesiti n_iter po zelji, to je broj kombinacija koje ce da se probaju

        random_search = clf.fit(X, y)

        print(random_search.best_params_)
        print(random_search.best_score_)
        print(random_search.cv_results_)

        file_path = 'data/results/results_'+str(model_name)+'.txt'

        with open(file_path, 'w') as file:
            file.write(str(random_search.best_params_) + '\n')
            file.write(str(random_search.best_score_) + '\n')
            file.write(str(random_search.cv_results_) + '\n')