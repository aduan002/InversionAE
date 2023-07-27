import os 
import pandas as pd
import numpy as np
import argparse
import yaml 
from tqdm import tqdm

from scipy import spatial

ZERO = 1e-9

def adjust_errors(errors, neighbors):
    zero_indices = np.where(errors <= ZERO)[0]

    index_value = {}
    for error_idx in zero_indices:
        new_error = np.mean(errors[neighbors[error_idx]])
        index_value[error_idx] = new_error

    for error_idx, new_error in index_value.items():
        errors[error_idx] = new_error
    return errors


def main(config):
    spatial_data  = config["spatial_data"]
    input_dir = config["input_dir"]
    save_dir = config["save_dir"]
    num_neighbors = config["num_neighbors"]

    row_data = pd.read_csv(spatial_data, delim_whitespace = True, names = ["row", "x", "y", "z"])
    row_data.sort_values("row", ascending=True, inplace=True)
    row_data.reset_index(inplace=True)

    coordinates = np.array(tuple(zip(row_data["x"], row_data["y"], row_data["z"])))
    kd_tree = spatial.cKDTree(coordinates)
    neighbors = kd_tree.query(coordinates, k=num_neighbors + 1)[1][:, 1:] # Get the indices for the k nearest neighbors, ignore the first value that refers to itself.

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = os.listdir(input_dir)
    print("Converting files:")
    for filename in tqdm(filenames):
        if os.path.isfile(os.path.join(save_dir, filename.split(".")[0] + ".obj")):
            continue
        
        errors = np.loadtxt(os.path.join(input_dir, filename), dtype=np.float32)
        errors = adjust_errors(errors, neighbors)
        
        with open(os.path.join(save_dir, filename.split(".")[0] + ".obj"), "w") as file:
            line = "v {0} {1} {2} {3}\n"
            for index, row in row_data.iterrows():
                x = row["x"]
                y = row["y"]
                z = row["z"]
                w = errors[index]

                file.write(line.format(x,y,z,w))

                # NOTE: No faces, just points...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Sigma to Obj Converter",
         description = "Convert Sigma predictions to Obj file format"
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as file:
        try:
            yaml_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    main(yaml_config)

    