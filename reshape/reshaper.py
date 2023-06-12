import os
import pandas as pd
import pickle
from sklearn.preprocessing import OrdinalEncoder

class FileReshape:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

        self.coordinates = {} # x,y,z coordinates for the ith row, but not all rows are present here.

        self.encoder = OrdinalEncoder(dtype=int)


    def reshape(self):
        # NOTE: Making two copies of the data, but it is constant, small size, and faster than iterating twice
        file_data = pd.read_csv(self.file_path, delim_whitespace = True, names = ["row", "x", "y", "z"])
        
        self.encoder.fit(file_data[["x", "y", "z"]])

        order_file_data = file_data.copy()
        order_file_data[["x", "y", "z"]] = self.encoder.transform(file_data[["x", "y", "z"]])

        for i in range(len(file_data)):
            row_number = file_data.iloc[i]["row"]

            raw_x = file_data.iloc[i]["x"]
            raw_y = file_data.iloc[i]["y"]
            raw_z = file_data.iloc[i]["z"]

            order_x = order_file_data.iloc[i]["x"]
            order_y = order_file_data.iloc[i]["y"]
            order_z = order_file_data.iloc[i]["z"]

            self.coordinates[row_number] = {
                "raw": {
                    "x": raw_x,
                    "y": raw_y,
                    "z": raw_z
                },
                "order": {
                    "x": order_x,
                    "y": order_y,
                    "z": order_z
                }
            }

    def save(self, file_dir, file_name):
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)

        save_content = {
            "encoder": self.encoder,
            "coordinates": self.coordinates
        }

        with open(os.path.join(file_dir, file_name), "wb") as file:
            pickle.dump(save_content, file, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, file_path):
        with open(file_path, "rb") as file:
            save_content = pickle.load(file)

        self.encoder = save_content["encoder"]
        self.coordinates = save_content["coordinates"]



if __name__ == "__main__":
    file_dir = "Reshapes"
    file_name = "spatial.pickle"

    reshaper = FileReshape("spatial.txt")

    reshaper.reshape()
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