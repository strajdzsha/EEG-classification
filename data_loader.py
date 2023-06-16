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

    def __init__(self, path: str, participants_ids: List, batch_size: int = 1, seed: int = None):
        
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
        if batch_size:
            self.batch_size = batch_size

        if seed:
            random.seed(seed)
    
        for i, fpath in enumerate(self.filepaths):
            self.filepaths[i] = fpath.replace('\\', '/')

        random.shuffle(self.filepaths)      
        self.idx = 0

    def __iter__(self):
        """
        Returns an iterator over the data
        """
        return self
    
    def __next__(self):
        """
        Returns the next epoch or batch of epochs if batch_size > 1
        """      
        if self.batch_size == 1:
            return self.nextEpoch()
        else:
            return [self.nextEpoch() for _ in range(self.batch_size)]

    def nextEpoch(self):
        """
        Returns the next epoch
        """   

        group_to_int = {'C': 0, 'A': 1, 'F': 2}

        if self.idx >= len(self.filepaths):
            raise StopIteration
        else:
            with open(self.filepaths[self.idx], 'rb') as f:
                data = pickle.load(f)
                
                folder_name = self.filepaths[self.idx].split('/')[-2]
                par_id, gender, age, group, mmse = folder_name.split('-')[1:]
                self.idx += 1
                return {
                    'par_id': int(par_id),
                    'data': data,
                    'gender': gender,
                    'age': int(age),
                    'group': group_to_int[group],
                    'mmse': int(mmse)
                }
                   
    def __getitem__(self, idx):
        """
        Returns the epoch at the given index
        If idx is of type slice, it returns all epochs
        in the given range
        """

        group_to_int = {'C': 0, 'A': 1, 'F': 2}

        if isinstance(idx, int):
            with open(self.filepaths[idx], 'rb') as f:
                data = pickle.load(f)
                folder_name = self.filepaths[idx].split('/')[-2]
                gender, age, group, mmse = folder_name.split('-')[2:]

                return {
                    'data': data,
                    'gender': gender,
                    'age': int(age),
                    'group': group_to_int[group],
                    'mmse': int(mmse)
                }
            
        elif isinstance(idx, slice):
            if idx.start is None:
                idx = slice(0, idx.stop, idx.step)
            if idx.stop is None:
                idx = slice(idx.start, len(self.filepaths), idx.step)
            if idx.step is None:
                idx = slice(idx.start, idx.stop, 1)
            return [self[i] for i in range(idx.start, idx.stop, idx.step)]

    def reset_iter(self):
        """
        Resets the iterator
        """
        self.idx = 0

    def __len__(self):
        """
        Returns the length of the dataset
        """
        return len(self.filepaths)


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

    train_loader = DataLoader(data_path, participants_ids = train_ids, batch_size=2)