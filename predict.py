import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import numpy as np
import os
from scipy.stats import norm
import math
import json 

from transformations.reshaper import FileSpatialReshaper
from transformations.scaler import CustomStandardScaler
from dataset import InversionDataset
from model import AutoEncoder

def percent_error(y, y_hat):
    difference = torch.abs(y - y_hat)
    exact = torch.abs(y)
    percent_error = difference / exact * 100
    return percent_error

def main(config, hyp):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = config["data"]
    save_dir = config["save_dir"]
    scaler_path = config["scaler"]
    reshaper_path = config["reshaper"]
    weights_path = config["weights"]
    upper_bounds_path = config["upper_bounds_path"]

    kernel_size = (hyp["kernel_size"]["depth"], hyp["kernel_size"]["height"], hyp["kernel_size"]["width"])
    stride = hyp["stride"]
    padding = hyp["padding"]
    num_hidden_layers = hyp["num_hidden_layers"]


    reshaper = None
    if reshaper_path is not None:
        reshaper = FileSpatialReshaper()
        reshaper.load(reshaper_path)

    scaler = None
    if scaler_path is not None:
        scaler = CustomStandardScaler()
        scaler.load(scaler_path)

    upper_bounds = None
    if upper_bounds_path is None:
        raise Exception("Error: Path to generated Upper Bounds is needed.")
    with open(upper_bounds_path, "rb") as file:
        upper_bounds = np.load(file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    data = InversionDataset(data_path, scaler=scaler, reshaper=reshaper)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    model = AutoEncoder(in_out_shape = data.__getitem__(0)[0].shape, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding, 
                        num_hidden_layers=num_hidden_layers)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        print("Calculating Percent Errors:")
        for batch, (tensor, filename) in enumerate(tqdm(dataloader)):
            filename = filename[0]  # for some reason, this returns the filename as (filename,) # TODO: Fix it.
            tensor = tensor.to(device)
            pred = model(tensor)
            tensor = tensor.to("cpu")

            orig_tensor = torch.from_numpy(scaler.inverse_transform(reshaper.inverse_transform(tensor)))
            orig_pred = torch.from_numpy(scaler.inverse_transform(reshaper.inverse_transform(pred)))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Convolutional AutoEncoder Prediction",
         description = "Predict Anomalies for a ConvolutionalAE"
    )
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-p", "--hyp", required=True)
    args = parser.parse_args()
    config_path = args.config
    hyp_path = args.hyp

    with open(config_path, "r") as file:
        try:
            yaml_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    with open(hyp_path, "r") as file:
        json_hyp = json.load(file)

    main(yaml_config, json_hyp)

    