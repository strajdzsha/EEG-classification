import os 
import pickle
import random
import numpy as np
from typing import List

def balanced_split(dataset_path: str, participant_ids: List[int] = None, num_test_part: int = 8):
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
    for group in group_ids.keys():
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


if __name__ == "__main__":
    participants_path = "./data/dataset"

    balanced_split(participants_path, list(range(88)))
