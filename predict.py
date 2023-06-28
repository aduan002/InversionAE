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
from transformations.scaler import PartialStandardScaler, FullStandardScaler
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
    aux_filepath = config["aux_data"]
    save_dir = config["save_dir"]
    main_scaler_path = config["main_scaler"]
    aux_scaler_path = config["aux_scaler"]
    reshaper_path = config["reshaper"]
    weights_path = config["weights"]
    upper_bounds_path = config["upper_bounds_path"]

    kernel_size = (hyp["kernel_size"]["depth"], hyp["kernel_size"]["height"], hyp["kernel_size"]["width"])
    stride = hyp["stride"]
    padding = hyp["padding"]
    conv_num_hidden_layers = hyp["conv_num_hidden_layers"]
    upsample_mode = hyp["upsample_mode"]

    lstm_hidden_size = hyp["lstm_hidden_size"]
    lstm_num_hidden_layers = hyp["lstm_num_hidden_layers"]
    lstm_dropout = hyp["lstm_dropout"]
    lstm_steps = hyp["lstm_steps"]


    reshaper = None
    if reshaper_path is not None:
        reshaper = FileSpatialReshaper()
        reshaper.load(reshaper_path)

    main_scaler = None
    if main_scaler_path is not None:
        main_scaler = PartialStandardScaler()
        main_scaler.load(main_scaler_path)
    aux_scaler = None
    if aux_scaler_path is not None:
        aux_scaler = FullStandardScaler()
        aux_scaler.load(aux_scaler_path)

    upper_bounds = None
    if upper_bounds_path is None:
        raise Exception("Error: Path to generated Upper Bounds is needed.")
    with open(upper_bounds_path, "rb") as file:
        upper_bounds = np.load(file)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    data = InversionDataset(data_path, aux_filepath, scaler=main_scaler, reshaper=reshaper, aux_scaler=aux_scaler, timesteps=lstm_steps, eval_mode=True)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    model = AutoEncoder(in_out_shape = data.__getitem__(0)[0].shape, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        padding=padding, 
                        conv_num_hidden_layers=conv_num_hidden_layers,
                        upsample_mode=upsample_mode,
                        lstm_input_shape=data.__getitem__(0)[1].shape,
                        lstm_hidden_size=lstm_hidden_size,
                        lstm_num_hidden_layers=lstm_num_hidden_layers,
                        lstm_dropout=lstm_dropout)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        print("Calculating Percent Errors:")
        for batch, (tensor, aux_tensor, filename) in enumerate(tqdm(dataloader)):
            tensor = tensor.to(device)
            pred = model(tensor, aux_tensor)
            tensor = tensor.to("cpu")

            orig_tensor = torch.from_numpy(main_scaler.inverse_transform(reshaper.inverse_transform(tensor)))
            orig_pred = torch.from_numpy(main_scaler.inverse_transform(reshaper.inverse_transform(pred)))

            errors = percent_error(orig_tensor, orig_pred).detach().cpu().numpy()

            for error in errors: # The batch size is 1, so not really needed at the moment.

                # NOTE: For now, assuming anomalies happen when values are unusually high and it is fine if they are unusually low... gotta ask about that
                delta = error - upper_bounds
                delta[delta < 0] = 0

                if np.any(delta):
                    print("Anomaly within {0}...".format(filename))

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

    