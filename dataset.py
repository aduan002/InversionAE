import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd

import re 
from datetime import datetime, timedelta

class InversionDataset(Dataset):
    def __init__(self, file_dir, aux_filepath, scaler = None, pca = None, reshaper = None,
                 aux_scaler = None, timesteps = 10,
                 aux_regex = r"([0-9]{4})([0-9]{2})([0-9]{2})_([0-9]{2})([0-9]{2})",
                 eval_mode = False) -> None:
        super().__init__()

        self.file_dir = file_dir
        self.file_names = sorted(os.listdir(self.file_dir))
        self.length = len(self.file_names)

        self.scaler = scaler
        self.pca = pca
        self.reshaper = reshaper
        self.aux_scaler = aux_scaler

        aux_data = pd.read_csv(aux_filepath)
        aux_data.columns = ["Timestamp", "Temperature C", "Temperature F", "Rainfall"]
        aux_data["Timestamp"] = pd.to_datetime(aux_data["Timestamp"], format="%m/%d/%Y %H:%M")
        aux_data.set_index("Timestamp", inplace=True)
        aux_data.drop(["Temperature C"], axis=1, inplace=True)

        if aux_scaler is not None and eval_mode is False:
            aux_data = pd.DataFrame(self.aux_scaler.fit_transform(aux_data), columns=aux_data.columns, index=aux_data.index)
        elif aux_scaler is not None and eval_mode is True:
            aux_data = pd.DataFrame(self.aux_scaler.transform(aux_data), columns=aux_data.columns, index=aux_data.index)

        self.aux_data = aux_data
        self.timesteps = timesteps
        self.aux_regex = aux_regex

    def __len__(self):
        return self.length
    
    def __getitem__(self, index) -> torch.tensor:
        file_name = self.file_names[index]
        item = np.loadtxt(os.path.join(self.file_dir, file_name), dtype=np.float32)

        # If scaler before PCA, it results in large value in -700s to 800s as well as close to 0 values...

        if self.pca is not None:
            item = self.pca.transform(item.reshape(1, -1))[0]
        if self.scaler is not None:
            # Reshape from [x_1, x_2, x_3, ..., x_n] into [ [x_1, x_2, x_3, ..., x_n] ] 
            # and then back into [x_1, x_2, x_3, ..., x_n]
            item = self.scaler.transform(item.reshape(1, -1))[0]
        if self.reshaper is not None:
            channels = 1
            depth = self.reshaper.size_z
            height = self.reshaper.size_y
            width = self.reshaper.size_x

            if depth is None or height is None or width is None:
                print("Error: depth {0}, height {1}, width {2}".format(depth, height, width))
                exit()

            # Conv3D takes input in the form of Batch, Channels, Depth, Height, Width
            shape_item = np.zeros((channels, depth, height, width), dtype=np.float32)

            idx = 0
            for row in sorted(self.reshaper.coordinates):
                coords = self.reshaper.coordinates[row]

                cell_value = item[idx]
                channel = 0
                z = coords["z"]
                y = coords["y"]
                x = coords["x"]

                shape_item[channel][z][y][x] = cell_value

                idx += 1
            item = shape_item


        year, month, day, hour, minute = re.findall(self.aux_regex, file_name)[0]

        current_time = datetime(int(year), int(month), int(day), int(hour), int(minute))
        # The closest time is the biggest time that is smaller or equal to the current time.
        closest_time = max(date for date in self.aux_data.index if date <= current_time)

        # This is the time granularity in the dataset
        delta = timedelta(minutes=15)

        # Get the last self.timesteps rows before the current time (and including the closest time)
        timesteps_time = self.aux_data.loc[closest_time - delta * (self.timesteps - 1): closest_time]
        weather_conditions = timesteps_time.to_numpy(dtype=np.float32)
        # Pad the beginning with zeros for the case where we don't have the time information self.timesteps rows back.
        weather_conditions = np.pad(weather_conditions, ((self.timesteps - weather_conditions.shape[0], 0),(0,0)))

        item = item.astype(np.float32)
        return torch.from_numpy(item), torch.from_numpy(weather_conditions), file_name