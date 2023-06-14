import os
import numpy as np
import pickle

from sklearn.decomposition import PCA

class CustomPCA:
    def __init__(self, kwargs = None) -> None:
        if kwargs is not None:
            self.pca = PCA(**kwargs)
        else:
            self.pca = PCA()

    def fit(self, file_dir, transformer = None):
        # NOTE: There is an IncrementalPCA with partial fit, but it is an approximation of PCA, so I would rather try to use the actual PCA if possible.
        full_data = None
        file_names = os.listdir(file_dir)

        idx = 0
        for file_name in file_names:
            data = np.loadtxt(os.path.join(file_dir, file_name), dtype=np.float32)

            if full_data is None:
                full_data = np.zeros((len(file_names), data.shape[0]))

            if transformer is not None:
                data = transformer.transform(data.reshape(1,-1))[0]

            full_data[idx] = data.copy()
            idx += 1
        self.pca.fit(full_data)

    def transform(self, X):
        return self.pca.transform(X)
    
    def inverse_transform(self, X):
        return self.pca.inverse_transform(X)
    
    def save(self, file_dir, file_name):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        with open(os.path.join(file_dir, file_name), "wb") as file:
            pickle.dump(self.pca, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            self.pca = pickle.load(file)

if __name__ == "__main__":
    file_dir = "data/train"

    pca = CustomPCA()

    pca.fit(file_dir)

    sample_file = os.listdir(file_dir)[0]
    X = np.loadtxt(os.path.join(file_dir, sample_file)).reshape(1, -1)
    pca_X = pca.transform(X)
    unpca_X = pca.inverse_transform(pca_X)

    print("X")
    print(X)
    print(X.shape)
    print("pca_X")
    print(pca_X.shape)
    print("unpca_X")
    print(unpca_X)

    pca.save("PCAs", "pca.pickle")
    pca.load(os.path.join("PCAs", "pca.pickle"))

    print("Loaded pca_X")
    print(pca.transform(X).shape)

    unpca_diff = X - unpca_X
    np.abs(unpca_diff, unpca_diff)
    print("unPCA difference sum:", unpca_diff.sum())



    
    