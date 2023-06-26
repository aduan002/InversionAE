import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import pandas as pd
import numpy as np
import os

from transformations.pca import CustomPCA
from transformations.scaler import PartialStandardScaler, FullStandardScaler
from dataset import InversionDataset
from model import AutoEncoder

def percent_error(y, y_hat):
    difference = torch.abs(y - y_hat)
    difference[difference<0] = 0  # y-y_hat<0 implies low conductivty anomaly... not anomaly
    exact = torch.abs(y)
    percent_error = difference / exact * 100
    return percent_error

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = config["data"]
    save_dir = config["save_dir"]
    scaler_path = config["scaler"]
    pca_path = config["pca"]
    weights_path = config["weights"]
    upper_bounds_path = config["upper_bounds_path"]

    pca = None
    if pca_path is not None:
        pca = CustomPCA()
        pca.load(pca_path)

    scaler = None
    if scaler_path is not None:
        scaler = PartialStandardScaler()
        scaler.load(scaler_path)

    upper_bounds = None
    if upper_bounds_path is None:
        raise Exception("Error: Path to generated Upper Bounds is needed.")
    with open(upper_bounds_path, "rb") as file:
        upper_bounds = np.load(file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = InversionDataset(data_path, scaler=scaler, pca=pca)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    model = AutoEncoder(in_out_shape = data.__getitem__(0)[0].shape)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    stats = pd.DataFrame(columns=["Filename", "Normal Count", "Anomalous Count"])

    with torch.no_grad():
        for batch, (tensor, filename) in enumerate(tqdm(dataloader)):
            filename = filename[0] # for some reason, this returns the filename as (filename,) # TODO: Fix it.
            tensor = tensor.to(device)
            pred = model(tensor)
            tensor = tensor.to("cpu")

            orig_tensor = torch.from_numpy(pca.inverse_transform(scaler.inverse_transform(tensor)))
            orig_pred = torch.from_numpy(pca.inverse_transform(scaler.inverse_transform(pred)))

            errors = percent_error(orig_tensor, orig_pred).detach().cpu().numpy()

            for error in errors: # The batch size is 1, so not really needed at the moment.

                # NOTE: For now, assuming anomalies happen when values are unusually high and it is fine if they are unusually low... gotta ask about that
                delta = error - upper_bounds
                delta[delta < 0] = 0

                if np.any(delta):
                    #print("Anomaly within {0}...".format(filename))
                    pass

                save_name = filename.split(".")[0] + ".sig"
                np.savetxt(os.path.join(save_dir, save_name), delta, fmt="%s")

                normalCount = len(delta[delta==0])
                anomalousCount = len(delta[delta!=0])
                row = [filename, normalCount, anomalousCount]
                stats.loc[len(stats)] = row
        stats.to_csv("stats.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Linear AutoEncoder Prediction",
         description = "Testing an AutoEncoder using Linear Layers"
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

    