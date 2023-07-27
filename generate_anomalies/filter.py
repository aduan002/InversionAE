from tqdm import tqdm

import pandas as pd
import numpy as np
import os

class FilterData:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

        file_data = pd.read_csv(self.file_path, delim_whitespace = True, names = ["row", "x", "y", "z"])
        file_data.sort_values("row", ascending=True, inplace=True)

        self.index_filter = np.zeros((file_data["row"].shape), dtype=int)
        for i in range(len(file_data)):
            row_idx = file_data.iloc[i]["row"] - 1

            self.index_filter[i] = row_idx

    def filter(self, X):
        output_shape = (self.index_filter.shape[0], ) + X.shape[1:]
        #data = np.empty((self.index_filter.shape), dtype="<U25")
        data = np.empty((output_shape), dtype="<U25")

        for idx, row_idx in enumerate(self.index_filter):
            data[idx] = X[row_idx]
        return data
    
    def filter_dataset(self, input_dir, output_dir):
        file_names = os.listdir(input_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for file_name in tqdm(file_names):
            data = np.loadtxt(os.path.join(input_dir, file_name), dtype="<U25") # The actual needed one is <U14

            filtered_data = self.filter(data)

            np.savetxt(os.path.join(output_dir, file_name), filtered_data, fmt="%s")

if __name__ == "__main__":
    file_path = "aux_data/spatial.txt"
    input_dir = "test_ground_truth_seed=1_factor=1.3_dist=1"
    #input_dir = "train_ground_truth_seed=0_dist=20"
    output_dir = "filtered_test_ground_truth_seed=1_factor=1.3_dist=1"
    #output_dir = "filtered_train_ground_truth_seed=0_dist=20"

    data_filter = FilterData(file_path)
    data_filter.filter_dataset(input_dir, output_dir)
    