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

# NOTE: This code is from https://github.com/jg-854/tolerance_intervals
def find_nearest(array, value):
    # nearest neighbour interpolation to find the input value which maps to the CDF's output value
    idx = (np.abs(array - value)).argmin()
    return idx
def one_sided_tolerance(alpha, beta, dataset, bootstrap_iterations=100):
    num_samples = len(dataset)

    # these are the empirical mean and std from the given sample, as opposed to the bootstrapped values seen later
    sample_mu = dataset.mean()
    sample_std = dataset.std()

    zp = norm.ppf(beta)
    za = norm.ppf(alpha)
    a = 1 - 0.5 * za * za / (num_samples - 1)
    b = zp * zp - za * za / num_samples

    k = (zp + math.pow(zp * zp - a * b, 0.5)) / a

    # these are the tolerance intervals ASSUMING our data is normally distributed
    upper_bound = sample_mu + (k * sample_std)

    dataset.sort()  # this forms an empirical cdf of the original dataset

    p = np.linspace(0, 0.995, num_samples)  # this generates the probability (y axis) for the cdf

    d = []
    for i in range(bootstrap_iterations):  # value is arbitrary: the higher the better the accuracy
        # the mean and std are calculated for the bootstrapped sample
        bsample = []
        for i in range(num_samples):
            bsample.append(np.random.choice(dataset, replace=True))
        bsample = np.asarray(bsample)
        bmu = bsample.mean()
        bstd = bsample.std()
        bsample.sort()

        f_sam_upper = p[find_nearest(bsample, bmu + (k * bstd))]
        f_emp_upper = p[find_nearest(dataset, bmu + (k * bstd))]

        db = math.pow(num_samples, 0.5) * (f_sam_upper - f_emp_upper)
        d.append(db)

    pcnt = np.percentile(d, alpha * 100)
    updated_beta = p[find_nearest(dataset, upper_bound)] - pcnt / math.sqrt(num_samples)

    return updated_beta, upper_bound, sample_mu, sample_std, num_samples

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

    kernel_size = (hyp["kernel_size"]["depth"], hyp["kernel_size"]["height"], hyp["kernel_size"]["width"])
    stride = hyp["stride"]
    padding = hyp["padding"]
    conv_num_hidden_layers = hyp["conv_num_hidden_layers"]
    upsample_mode = hyp["upsample_mode"]

    lstm_hidden_size = hyp["lstm_hidden_size"]
    lstm_num_hidden_layers = hyp["lstm_num_hidden_layers"]
    lstm_dropout = hyp["lstm_dropout"]

    confidence = config["confidence"]
    coverage = config["coverage"]

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

    data = InversionDataset(data_path, aux_filepath, scaler=main_scaler, reshaper=reshaper, aux_scaler=aux_scaler, eval_mode=True)
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

    samples = None
    idx = 0
    with torch.no_grad():
        print("Calculating Percent Errors:")
        for batch, (tensor, aux_tensor, _) in enumerate(tqdm(dataloader)):
            tensor = tensor.to(device)
            pred = model(tensor, aux_tensor)
            tensor = tensor.to("cpu")

            orig_tensor = torch.from_numpy(main_scaler.inverse_transform(reshaper.inverse_transform(tensor)))
            orig_pred = torch.from_numpy(main_scaler.inverse_transform(reshaper.inverse_transform(pred)))

            errors = percent_error(orig_tensor, orig_pred).detach().cpu().numpy()

            for error in errors: # The batch size is 1, so not really needed at the moment.

                if samples is None:
                    samples = np.zeros((len(data), error.shape[0]))
                samples[idx] = error
                idx += 1

    upper_bounds = np.zeros((samples.shape[1]))
    print("Calculating Upper Bounds:")
    for idx in tqdm(range(samples.shape[1])):
        updated_beta, upper_bound, sample_mu, sample_std, num_samples = one_sided_tolerance(confidence, coverage, samples[:,idx], bootstrap_iterations=100)

        upper_bounds[idx] = upper_bound

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Saving a npy because performance is much better than txt (or csv) and I don't care about human readability.
    save_name = "upper_bounds-conf_{0}-cov_{1}.npy".format(confidence, coverage)
    with open(os.path.join(save_dir, save_name), "wb") as file:
        np.save(file, upper_bounds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Convolutional AutoEncoder Threshold Generation",
         description = "Generate Upper Bounded Tolerance Intervals for a ConvolutionalAE"
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

    