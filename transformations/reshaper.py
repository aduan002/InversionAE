import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OrdinalEncoder

class FileSpatialReshaper:
    def __init__(self, file_path = None) -> None:
        # NOTE: Filepath should only be "None" when loading a reshaper, not building it.
        self.file_path = file_path

        self.coordinates = {} # x,y,z coordinates for the ith row, but not all rows are present here.
        self.encoder = OrdinalEncoder(dtype=int)

        self.size_x = None
        self.size_y = None
        self.size_z = None


    def build_reshape(self):
        # NOTE: Making two copies of the data, but it is constant, small size, and faster than iterating twice
        file_data = pd.read_csv(self.file_path, delim_whitespace = True, names = ["row", "x", "y", "z"])
        self.encoder.fit(file_data[["z"]])

        self.size_z = len(self.encoder.categories_[0])

        order_file_data = file_data.copy()
        del file_data # The copy was done when I was saving the raw and ordered coordinates together, so for now, free space.

        order_file_data[["z"]] = self.encoder.transform(order_file_data[["z"]])

        order_file_data = order_file_data.to_numpy(dtype=np.float64)
        row_col = 0
        x_col = 1
        y_col = 2
        z_col = 3

        for z in range(self.size_z):
            z_indices = np.where(order_file_data[:, z_col] == z)[0]
            sorted_z_indices = z_indices[np.argsort(order_file_data[z_indices, y_col], kind="stable")]

            y = 0
            for z_idx in sorted_z_indices:
                order_file_data[z_idx][y_col] = y

                y_indices = np.where(order_file_data[:, y_col] == y)[0]
                zy_indices = np.intersect1d(sorted_z_indices, y_indices)

                x_indices = zy_indices[np.argsort(order_file_data[zy_indices, x_col], kind="stable")]
                x = 0
                for x_idx in x_indices:
                    order_file_data[x_idx][x_col] = x
                    x += 1

                y += 1
      
        order_file_data = order_file_data.astype(dtype=int)
        max_values = np.amax(order_file_data, axis=0)
        self.size_y = max_values[y_col] + 1
        self.size_x = max_values[x_col] + 1


        for i in range(len(order_file_data)):
            row_number = order_file_data[i][row_col] - 1

            order_x = order_file_data[i][1] 
            order_y = order_file_data[i][2] 
            order_z = order_file_data[i][3] 

            self.coordinates[row_number] = {
                "x": order_x,
                "y": order_y,
                "z": order_z
            }
            
    # Using numpy is at least 4 times faster... but keeping this method for now in case some of its code is needed.
    def build_reshape_pandas(self):
        # NOTE: Making two copies of the data, but it is constant, small size, and faster than iterating twice
        file_data = pd.read_csv(self.file_path, delim_whitespace = True, names = ["row", "x", "y", "z"])

        self.encoder.fit(file_data[["z"]])

        self.size_z = len(self.encoder.categories_[0])

        order_file_data = file_data.copy()
        del file_data # The copy was done when I was saving the raw and ordered coordinates together, so for now, free space.

        order_file_data[["z"]] = self.encoder.transform(order_file_data[["z"]])

        for z in range(self.size_z):
            y_indices = order_file_data[order_file_data["z"] == z].sort_values("y", ascending=True, kind="stable").index
            y = 0
            for y_idx in y_indices:
                order_file_data.loc[y_idx, "y"] = y

                y_order_file_data = order_file_data.iloc[y_indices]

                x_indices = y_order_file_data[y_order_file_data["y"] == y].sort_values("x", ascending=True, kind="stable").index
                x = 0
                for x_idx in x_indices:
                    order_file_data.loc[x_idx, "x"] = x
                    x += 1

                y += 1
        order_file_data["y"] = order_file_data["y"].astype("int")
        self.size_y = max(order_file_data["y"]) + 1
        order_file_data["x"] = order_file_data["x"].astype("int")
        self.size_x = max(order_file_data["x"]) + 1
        

        # NOTE: No duplicates, so that's good.
        #duplicates = order_file_data.duplicated(subset=["x","y","z"]).any()
        #print("Duplicates:", duplicates)

        order_file_data.sort_values("row", ascending=True, inplace=True)
        for i in range(len(order_file_data)):
            row_number = order_file_data.iloc[i]["row"] - 1 # rows start at 1 -> zero index rows

            order_x = order_file_data.iloc[i]["x"]
            order_y = order_file_data.iloc[i]["y"]
            order_z = order_file_data.iloc[i]["z"]

            self.coordinates[row_number] = {
                "x": order_x,
                "y": order_y,
                "z": order_z
            }


    def inverse_transform(self, shaped_tensors):
        orig_tensors = np.zeros((shaped_tensors.shape[0], len(self.coordinates)))

        for batch_idx in range(len(shaped_tensors)):
            shaped_tensor = shaped_tensors[batch_idx]
            orig_tensor = np.zeros((len(self.coordinates)))

            idx = 0
            for row in sorted(self.coordinates):
                coords = self.coordinates[row]

                channel = 0
                z = coords["z"]
                y = coords["y"]
                x = coords["x"]
                cell_value = shaped_tensor[channel][z][y][x]

                orig_tensor[idx] = cell_value
                idx += 1

            orig_tensors[batch_idx] = orig_tensor
        return orig_tensors

    def save(self, file_dir, file_name):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        save_content = {
            "encoder": self.encoder,
            "coordinates": self.coordinates,
            "size_x": self.size_x,
            "size_y": self.size_y,
            "size_z": self.size_z
        }

        with open(os.path.join(file_dir, file_name), "wb") as file:
            pickle.dump(save_content, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            save_content = pickle.load(file)

        self.encoder = save_content["encoder"]
        self.coordinates = save_content["coordinates"]
        self.size_x = save_content["size_x"]
        self.size_y = save_content["size_y"]
        self.size_z = save_content["size_z"]



if __name__ == "__main__":
    file_dir = "filtered_data/train"
    file_name = "FA3_8L20220928_2001.sig"

    reshaper = FileSpatialReshaper("aux_data/spatial.txt")
    reshaper.build_reshape()
    reshaper.save("Reshapers", "spatial_reshape.pickle")

    channels = 1
    depth = reshaper.size_z
    height = reshaper.size_y
    width = reshaper.size_x

    item = np.loadtxt(os.path.join(file_dir, file_name), dtype=np.float32)
    reconstruction_item = np.zeros((item.shape), dtype=np.float32)

    shape_item = np.zeros((channels, depth, height, width), dtype=np.float32)
    idx = 0
    for row in sorted(reshaper.coordinates):
        coords = reshaper.coordinates[row]

        cell_value = item[idx]
        channel = 0
        z = coords["z"]
        y = coords["y"]
        x = coords["x"]

        shape_item[channel][z][y][x] = cell_value

        idx += 1

    # NOTE: This is how you do the reconstruction
    idx = 0
    for row in sorted(reshaper.coordinates):
        coords = reshaper.coordinates[row]

        channel = 0
        z = coords["z"]
        y = coords["y"]
        x = coords["x"]
        cell_value = shape_item[channel][z][y][x]

        reconstruction_item[idx] = cell_value
        idx += 1

    print("#####################################################")
    print((item == reconstruction_item).all())
    print("#####################################################")











    # NOTE: Below here there are some previous test cases.
    exit()
    
    import time

    file_dir = "Reshapes"
    file_name = "spatial.pickle"

    reshaper = FileSpatialReshape("aux_data/spatial.txt")
    pd_start = time.time()
    reshaper.build_reshape()
    pd_end = time.time()
    pd_coords = reshaper.coordinates.copy()
    #print("######################################################################")
    #print("pd_coords")
    #print(pd_coords)
    #print("######################################################################")

    reshaper = FileSpatialReshape("aux_data/spatial.txt")
    np_start = time.time()
    reshaper.build_reshape_2()
    np_end = time.time()
    np_coords = reshaper.coordinates.copy()
    #print("######################################################################")
    #print("np_coords")
    #print(np_coords)
    #print("######################################################################")

    print("Equal?:", pd_coords == np_coords)
    i = 0
    print("Keys:", len(np_coords.keys()) == len(pd_coords.keys()))
    idx = 0
    for key in pd_coords:
        pd_v = pd_coords[key]
        np_v = np_coords[key]

        if pd_v != np_v:
            print("Not equal")
            print("idx:", idx)
            print("key:", key)
            break
        idx += 1
    for key in np_coords:
        pd_v = pd_coords[key]
        np_v = np_coords[key]

        if pd_v != np_v:
            print("Not equal")
            print(key)
            break
    print("pd duration: {0} minutes".format((pd_end - pd_start) / 60))
    print("np duration: {0} minutes".format((np_end - np_start) / 60))

    """
    print("#############################")
    print(reshaper.encoder)
    print(dict(list(reshaper.coordinates.items())[:10]))
    print("#############################")

    reshaper.save(file_dir, file_name)

    reshaper.encoder = None
    reshaper.coordinates = {}

    reshaper.load(os.path.join(file_dir, file_name))

    print("#############################")
    print(reshaper.encoder)
    print(dict(list(reshaper.coordinates.items())[:10]))
    print("#############################")
    """

    """
    # NOTE: This seems to indicate a very sparse 3D matrix where there are a very small number accesses for a given axis.
    # So, if we just use the OrdinalEncoder on x,y,z 
    order_x_list = [0 for _ in range(reshaper.size_x)]
    order_y_list = [0 for _ in range(reshaper.size_y)]
    order_z_list = [0 for _ in range(reshaper.size_z)]

    for row in sorted(reshaper.coordinates):
        coords = reshaper.coordinates[row]
        x = coords["x"]
        y = coords["y"]
        z = coords["z"]

        order_x_list[x] += 1
        order_y_list[y] += 1
        order_z_list[z] += 1
    print("#####################################################################################")
    print("order_x_list:", [idx for idx, value in enumerate(order_x_list) if value == 0])
    print("order_y_list:", [idx for idx, value in enumerate(order_y_list) if value == 0])
    print("order_z_list:", [idx for idx, value in enumerate(order_z_list) if value == 0])
    print("######################################################################################")
    """
    print("################################################################")
    print("reshaper.size_x:", reshaper.size_x)
    print("reshaper.size_y:", reshaper.size_y)
    print("reshaper.size_z:", reshaper.size_z)
    print("################################################################")
    
    