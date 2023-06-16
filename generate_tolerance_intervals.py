import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import yaml
import pandas as pd
import numpy as np
import os
from scipy.stats import norm
import math

from pca import CustomPCA
from scaler import CustomStandardScaler
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
    pca_path = config["pca"]
    weights_path = config["weights"]

    confidence = config["confidence"]
    coverage = config["coverage"]

    pca = None
    if pca_path is not None:
        pca = CustomPCA()
        pca.load(pca_path)

    scaler = None
    if scaler_path is not None:
        scaler = CustomStandardScaler()
        scaler.load(scaler_path)

    data = InversionDataset(data_path, scaler=scaler, pca=pca)
    dataloader = DataLoader(data, batch_size=1, shuffle=False)

    model = AutoEncoder(in_out_shape = data.__getitem__(0)[0].shape)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    # Source: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    # Welford's online algorithm
    stream_mean = None
    stream_M2 = None
    count = 0

    samples = None
    idx = 0
    with torch.no_grad():
        print("Calculating Percent Errors:")
        for batch, (tensor, _) in enumerate(tqdm(dataloader)):
            tensor = tensor.to(device)
            pred = model(tensor)
            tensor = tensor.to("cpu")

            orig_tensor = torch.from_numpy(pca.inverse_transform(scaler.inverse_transform(tensor)))
            orig_pred = torch.from_numpy(pca.inverse_transform(scaler.inverse_transform(pred)))

            errors = percent_error(orig_tensor, orig_pred).detach().cpu().numpy()

            for error in errors: # The batch size is 1, so not really needed at the moment.

                if samples is None:
                    samples = np.zeros((len(data), error.shape[0]))
                samples[idx] = error
                idx += 1

    samples = samples[:,:1000]
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
         prog = "Linear AutoEncoder Test",
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

    