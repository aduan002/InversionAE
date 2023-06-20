import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import numpy as np
import os
from scipy.stats import norm
import math

from transformations.reshaper import FileSpatialReshaper
from transformations.scaler import CustomStandardScaler
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

def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_path = config["data"]
    save_dir = config["save_dir"]
    scaler_path = config["scaler"]
    reshaper_path = config["reshaper"]
    weights_path = config["weights"]
    upper_bounds_path = config["upper_bounds_path"]


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

    model = AutoEncoder(in_out_shape = data.__getitem__(0)[0].shape)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        print("Calculating Percent Errors:")
        for batch, (tensor, filename) in enumerate(tqdm(dataloader)):
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
                    print("Anomaly within {0}...".format(filename))

                save_name = filename.split(".")[0] + ".sig"
                np.savetxt(os.path.join(save_dir, save_name), delta, fmt="%s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Convolutional AutoEncoder Prediction",
         description = "Predict Anomalies for a ConvolutionalAE"
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

    