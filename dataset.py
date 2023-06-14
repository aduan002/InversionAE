import torch
from torch.utils.data import Dataset
import os
import numpy as np

class InversionDataset(Dataset):
    def __init__(self, file_dir, scaler = None, pca = None, reshaper = None) -> None:
        super().__init__()

        self.file_dir = file_dir
        self.file_names = sorted(os.listdir(self.file_dir))
        self.length = len(self.file_names)

        self.scaler = scaler
        self.pca = pca
        self.reshaper = reshaper

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

        
        item = item.astype(np.float32)
        return torch.from_numpy(item)