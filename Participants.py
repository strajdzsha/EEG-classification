import numpy as np
import os
import pickle


class Participants:
    def __init__(self, data_path: str):
        """
        data_path (str) - path to the data folder (.../participants)
        """
        self.data_path = data_path
        self.train_ids = None # list of participant ids used for training
        self.test_ids = None # list of participant ids used for testing

        all_subjects = [x for x in os.listdir(data_path) if x.startswith('sub-')]
        self.name = [None] * len(all_subjects)
        
        for name in all_subjects:
            self.name[int(name.split('-')[1])] = name

        print(self.name[0])

    def __getitem__(self, idx):
        """
        idx (int) - index of the participant to load
        Returns:
            data (np.ndarray time series)
            gender (str) - 'M' or 'F'
            age (int) - age of the participant
            group (str) - 'A', 'C' or 'F'
            mmse (int) - mini mental state examination score
        """
        # load the data
        with open(os.path.join(self.data_path, self.name[idx]), 'rb') as f:
            data = pickle.load(f)
            name = self.name[idx].split('.')[0] # without extension
            _, _, gender, age, group, mmse = name.split('-')

        return data, gender, int(age), group, int(mmse)
        
    

if __name__ == "__main__":
    data_path = "./data/participants"
    participants = Participants(data_path)
    
    print(type(participants[0][4]))