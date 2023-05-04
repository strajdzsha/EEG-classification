import numpy as np
import random
import os
import pickle
from typing import List

class DataLoader:
    """
    Class used for loading the training or 
    testing data
    """
    fs = 500 # for our database this is fixed

    def __init__(self, path: str, participants_ids: List, seed: int = None):
        
        assert os.path.exists(path)
        self.path = path

        self.filepaths = []

        for folder_name in os.listdir(path):
            if not folder_name.startswith('sub-'):
                continue
            idx = int(folder_name.split('-')[1])
            
            # save paths to iterate over them
            folder_path = os.path.join(path, folder_name)
            if idx in participants_ids:
                for file_name in os.listdir(folder_path):
                    if not file_name.startswith('ep_'):
                        continue
                    self.filepaths.append(os.path.join(folder_path, file_name))

        if seed:
            random.seed(seed)
    
        random.shuffle(self.filepaths)      
        self.idx = 0

    def __iter__(self):
        """
        Returns an iterator over the data
        """
        return self
    
    def __next__(self):
        """
        Returns the next epoch
        """
        if self.idx >= len(self.filepaths):
            raise StopIteration
        else:
            with open(self.filepaths[self.idx], 'rb') as f:
                data = pickle.load(f)
                folder_name = self.filepaths[self.idx].split('/')[-2]
                gender, age, group, mmse = folder_name.split('-')[2:]
                self.idx += 1
                return {
                    'data': data,
                    'gender': gender,
                    'age': int(age),
                    'group': group,
                    'mmse': int(mmse)
                }

    def reset_iter(self):
        """
        Resets the iterator
        """
        self.idx = 0


if __name__ == "__main__":

    """
    Example usage
    using data loaders for training and testing
    """

    data_path = "./data/dataset"
    participant_ids = list(range(88))
    random.shuffle(participant_ids)

    train_ids = participant_ids[:80]
    test_ids = participant_ids[80:]

    train_loader = DataLoader(data_path, train_ids)
    test_loader = DataLoader(data_path, test_ids)

    for x in train_loader:
        
        # get data and labels
        data = x['data']
        group = x['group']
        mmse = x['mmse']
        age = x['age']
        gender = x['gender']

        # extract features
        pass

    for x in test_loader:
        # do the same for the test loader
        pass