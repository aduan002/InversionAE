import os
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler

class CustomStandardScaler:
    def __init__(self, transformer = None) -> None:
        self.scaler = StandardScaler(with_mean=False, with_std=False)

        self.transformer = transformer

    def fit(self, file_dir):
        file_names = os.listdir(file_dir)

        for file_name in file_names:
            data = np.loadtxt(os.path.join(file_dir, file_name), dtype=np.float32)

            if self.transformer is not None:
                data = self.transformer.transform(data.reshape(1,-1))[0]

            self.scaler.partial_fit(data.reshape(1, -1))  # Reshape takes [x_1, x_2, x_3, ..., x_n] into [ [x_1, x_2, x_3, ..., x_n] ]

    def transform(self, X):
        return self.scaler.transform(X)
    
    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)
    
    def save(self, file_dir, file_name):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        with open(os.path.join(file_dir, file_name), "wb") as file:
            pickle.dump(self.scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            self.scaler = pickle.load(file)

if __name__ == "__main__":
    file_dir = "data/train"

    scaler = CustomStandardScaler()

    scaler.fit(file_dir)

    sample_file = os.listdir(file_dir)[0]
    X = np.loadtxt(os.path.join(file_dir, sample_file)).reshape(1, -1)
    scaled_X = scaler.transform(X)
    unscaled_X = scaler.inverse_transform(scaled_X)

    print("X")
    print(X)
    print("scaled_X")
    print(scaled_X)
    print("unscaled_X")
    print(unscaled_X)

    scaler.save("scalers", "standard_scaler.pickle")
    scaler.load(os.path.join("scalers", "standard_scaler.pickle"))

    print("Loaded scaled_X")
    print(scaler.transform(X))

    
    