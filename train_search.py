import configparser
import numpy as np
import xgboost as xgb
import pandas as pd
import os 
import json

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from pycm import ConfusionMatrix

from utils import balanced_split, leave_one_out, get_git_root
from constants import RESULTS_PATH
import warnings

from temp import top_uncorrelated_features

warnings.filterwarnings("ignore")

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


def get_param_grid(model_name):
    if model_name == 'knn':
        param_grid = {'model__n_neighbors':[11, 7]}

    elif model_name == 'random_forest':
        param_grid = {'model__n_estimators':[100], 'model__max_depth':[4]}

    elif model_name == 'xgboost':
        param_grid = {'model__n_estimators':[100], 'model__max_depth':[4]}

    elif model_name == 'svm':
        param_grid = {'model__gamma': [1e-3, 1e-4], 'model__C': [1, 10, 100, 1000]}
    else:
        param_grid = {}

    return param_grid

if __name__ == "__main__":

    config_path = os.path.join(get_git_root(), 'config.ini')
    config = configparser.ConfigParser()
    config.read(config_path)

    n_folds = int(config['Train']['n_folds'])
    num_test_part = int(config['Train']['num_test_part'])
    n_features = [int(x.strip()) for x in config['Train']['n_features'].split(',')]
    model_names = [x.strip() for x in config['Model']['model_name'].split(',')]

    print("n_folds is {} and num_test_part is {}".format(n_folds, num_test_part))

    X = pd.read_csv(config['Data']['features_path'])
    print(config['Data']['features_path'])
    X.drop('Unnamed: 0', axis=1, inplace=True)

    outlier_ids = [] if config['Train']['use_outliers'] else [5, 40, 55, 25, 29, 31, 35, 85, 74, 24, 63, 79]

    # selecting features
    top_features = []
    with open('mutual_info.txt', 'r') as file:
        for line in file:
            ft = line.strip().split(' ')[0]
            top_features.append((line.strip()).split(' ')[0])

    for n in n_features:
        top_features = top_uncorrelated_features(X, n, top_features)
        #top_features = top_features[:n]
        X = X[top_features]

        par_ids = np.load(config['Data']['par_ids_path'])
        X['par_ids'] = par_ids

        y = np.load(config['Data']['labels_path'])
        X['labels'] = y
        X = X[X['labels'] != 2]

        for id in outlier_ids:
            X = X[X['par_ids'] != id]

        y = X['labels']
        X.drop(['labels'], axis=1, inplace=True)

        dataset_path = config['Data']['dataset_path']
        myCViterator = []

        for i in range(n_folds):
            train_ids, test_ids = balanced_split(dataset_path, num_test_part=num_test_part, two_classes=True)
            train_idx = np.where(np.array(X['par_ids'].isin(train_ids)) == True)[0]
            test_idx = np.where(np.array(X['par_ids'].isin(test_ids)) == True)[0]
            if len(test_idx) == 0: continue
            myCViterator.append((train_idx, test_idx))
        X = X.drop('par_ids', axis=1)

        
        for model_name in model_names:
            model = get_model(model_name)

            
            pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
            ])

            print(f"Trying model {model_name}")

            scoring = {
                'accuracy': make_scorer(accuracy_score)
            }

            param_grid = get_param_grid(model_name)

            clf = RandomizedSearchCV(pipeline, param_distributions=param_grid, cv = myCViterator, scoring=scoring, refit='accuracy', error_score="raise") # podesiti n_iter po zelji, to je broj kombinacija koje ce da se probaju

            random_search = clf.fit(X, y)

            if not os.path.exists(RESULTS_PATH):
                os.makedirs(RESULTS_PATH)

            out_path = os.path.join(RESULTS_PATH, f'results_{model_name}_{n}.json')
            
            out_json = {
                'mean_test_accuracy': list(random_search.cv_results_['mean_test_accuracy']),
                'std_test_accuracy': list(random_search.cv_results_['std_test_accuracy']),
                'best_params_': random_search.best_params_,
                'best_score_': random_search.best_score_
            }

            print(out_json)

            with open(out_path, 'w') as file:
                json.dump(out_json, file, indent=4)