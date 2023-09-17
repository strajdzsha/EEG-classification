import os 
import git
import pickle
import random
import numpy as np
from typing import List
from matplotlib import pyplot as plt

def get_n_participants(dataset_path: str):
    """
    Returns the number of participants in the dataset
    """
    return len(os.listdir(dataset_path))

def leave_one_out(dataset_path: str, par_id: int):
    train_ids = [x for x in range(88) if x != par_id]
    test_ids = [par_id]

    return train_ids, test_ids

def balanced_split(dataset_path: str, participant_ids: List[int] = None, num_test_part: int = 8, seed: int = None, two_classes = False):
    """
    Takes a list of participant ids and returns balanced split 
    to train and test sets, with adequate percentages of classes
    Returns:
        train_ids: List[int], test_ids: List[int]
    """
    if participant_ids is None:
        participant_ids = list(range(88))

    group_nums = {'C': 0, 'A': 0, 'F': 0}
    group_ids = {'C': [], 'A': [], 'F': []}
    for folder_name in os.listdir(dataset_path):
        if folder_name.startswith('sub-'):
            group = folder_name.split('-')[4]
            group_nums[group] += 1
            idx = int(folder_name.split('-')[1])
            group_ids[group].append(idx)
    
    assert num_test_part <= min(group_nums.values()) # if we want too many test participants

    train_ids = []
    test_ids = []
    groups = []
    if two_classes:groups = ['C', 'A'] # only use C and A groups
    else: groups = group_ids.keys() # use all groups
    for group in groups:
        if seed:
            random.seed(seed)
        if seed:
            random.seed(seed)
        random.shuffle(group_ids[group])
        test_ids += group_ids[group][:num_test_part]
        train_ids += group_ids[group][num_test_part:]
    
    return train_ids, test_ids


def parse_config_features(config):
    """
    Parses config file and returns a dictionary of features
    """
    feature_list = [
        k 
        for k, v in dict(config['Features']).items()
        if v == 'True'
    ] # only keep features that were selected as True in config
    pca_components = None
    if 'pca_n_components' in config['Features']:
        pca_components = int(config['Features']['pca_n_components'])

    all_features = {
        'features' : feature_list,
        'pca_components' : pca_components
    }
    return all_features
    
def shape_wrapper(data, func, shape):
    """
    Wrapper function for applying a function to a matrix
    with np.apply_along_axis (but along multiple axes)
    """
    data = data.reshape(shape)
    return func(data)

def plot_confusion_matrix(metrics):
    metrics.plot(cmap=plt.cm.Blues, number_label=True, plot_lib="seaborn")
    plt.show()

def get_git_root():
    repo = git.Repo('.', search_parent_directories=True)
    return repo.working_tree_dir


def top_uncorrelated_features(X, n_features, top_features):
    
    cnt = 0
    next_ft_idx = 0
    result_features = []
    while cnt < n_features:
        for ft in top_features[next_ft_idx:]:
            corr = X[result_features + [ft]].corr()
            mask = np.ones_like(corr, dtype=np.bool)
            mask[np.tril_indices_from(mask)] = False
            
            corr = corr * mask
            if corr[ft].max() < 0.5:
                result_features.append(ft)
                cnt += 1
                next_ft_idx = top_features.index(ft) + 1
                break
    
    return result_features

if __name__ == "__main__":
    path = get_git_root()
