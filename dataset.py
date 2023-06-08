import torch
from torch.utils.data import Dataset
import os
import numpy as np

class InversionDataset(Dataset):
    def __init__(self, file_dir, scaler = None) -> None:
        super().__init__()

        self.file_dir = file_dir
        self.file_names = sorted(os.listdir(self.file_dir))
        self.length = len(self.file_names)

        self.scaler = scaler

    def __len__(self):
        return self.length
    
    def __getitem__(self, index) -> torch.tensor:
        file_name = self.file_names[index]
        item = np.loadtxt(os.path.join(self.file_dir, file_name), dtype=np.float32)

        if self.scaler is not None:
            # Reshape from [x_1, x_2, x_3, ..., x_n] into [ [x_1, x_2, x_3, ..., x_n] ] 
            # and then back into [x_1, x_2, x_3, ..., x_n]
            item = self.scaler.transform(item.reshape(1, -1))[0]  

        return torch.from_numpy(item)