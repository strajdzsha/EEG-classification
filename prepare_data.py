import os
import pickle
import numpy as np
from Participants import Participants


participants = Participants("./data/participants")
output_path = "./data/dataset"

if not os.path.exists(output_path):
    os.mkdir(output_path)

fs = 500
for i in range(len(participants)):
    folder_name = participants.name[i].split('.')[0]
    curr_folder_path = os.path.join(output_path, folder_name)
    
    if not os.path.exists(curr_folder_path):
        os.mkdir(curr_folder_path)
    
    arr, gender, age, group, mmse = participants[i]
    interval = 5 * fs
    for j in range(len(arr[0]) // interval):
        curr_data = arr[:,j * interval: (j + 1) * interval]
        curr_path = os.path.join(curr_folder_path, f"ep_{j}.pickle")
        with open(curr_path, 'wb') as f:
            pickle.dump(curr_data, f)
