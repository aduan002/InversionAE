import os 
import pandas as pd
import numpy as np
import argparse
import yaml 
from tqdm import tqdm

def main(config):
    spatial_data  = config["spatial_data"]
    input_dir = config["input_dir"]
    save_dir = config["save_dir"]

    row_data = pd.read_csv(spatial_data, delim_whitespace = True, names = ["row", "x", "y", "z"])
    row_data.sort_values("row", ascending=True, inplace=True)
    row_data.reset_index(inplace=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filenames = os.listdir(input_dir)
    print("Converting files:")
    for filename in tqdm(filenames):
        errors = np.loadtxt(os.path.join(input_dir, filename), dtype=np.float32)

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

    