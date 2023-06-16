import torch
from torch.utils.data import DataLoader
import wandb
import argparse
import yaml
import json
from tqdm import tqdm
import os

from model import AutoEncoder
from dataset import InversionDataset
from scaler import CustomStandardScaler
from pca import CustomPCA

def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    for batch, (tensor, _) in enumerate(tqdm(dataloader)):
        tensor = tensor.to(device)
        # Compute prediction and loss
        pred = model(tensor)
        loss = loss_fn(pred, tensor)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.item()

    train_loss /= num_batches
    return train_loss

def val_loop(dataloader, model, loss_fn, device):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for batch, (tensor, _) in enumerate(tqdm(dataloader)):
            tensor = tensor.to(device)
            pred = model(tensor)
            loss = loss_fn(pred, tensor)

            val_loss += loss.item()

    val_loss /= num_batches
    return val_loss

def save_model(model, save_dir, save_name):
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))

def main(config, hyp):    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    train_dir = config["data"]["train"]
    val_dir = config["data"]["val"]

    batch_size = hyp["batch_size"]
    epochs = hyp["epochs"]
    learning_rate = hyp["learning_rate"]
    weight_decay = hyp["weight_decay"]
    patience = hyp["patience"]

    save_dir = config["model"]["save_dir"]
    save_freq = config["model"]["save_freq"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # PCA
    pca = CustomPCA()
    pca.fit(train_dir)
    pca.save("PCAs", "pca.pickle")

    # Scale
    scaler = CustomStandardScaler()
    scaler.fit(train_dir, pca)
    scaler.save("Scalers", "standard_scaler.pickle")

    train_data = InversionDataset(train_dir, scaler=scaler, pca=pca)
    val_data = InversionDataset(val_dir, scaler=scaler, pca=pca)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    model = AutoEncoder(in_out_shape = train_data.__getitem__(0)[0].shape)
    model.to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay)
    
    early_stop_count = 0
    prev_val_loss = None
    for e in range(1, epochs + 1):
        print("Epoch:", e)
        
        print("Training:")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer, device)
        
        print("Validation:")
        val_loss = val_loop(val_dataloader, model, loss_fn, device) 

        wandb.log({"train_mse_loss": train_loss, "val_mse_loss": val_loss})

        if e % save_freq == 0:
            save_name = str(e).zfill(len(str(epochs))) + ".pt"
            save_model(model, save_dir, save_name)

        if prev_val_loss is None:
            prev_val_loss = val_loss

        # The idea is that if the val loss has increased patience times in a row, stop training.
        delta = val_loss - prev_val_loss
        if delta > 0:
            early_stop_count += 1
        else:
            early_stop_count = 0

        prev_val_loss = val_loss

        if early_stop_count == patience:
            print("Early stopping after {0} epochs...".format(e))
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Linear AutoEncoder",
         description = "Training an AutoEncoder using Linear Layers"
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

    wandb.init(
        #mode="disabled",

        project="Linear AutoEncoder",
        name="Test Run 3",
        notes="Training Linear AutoEncoder on Inversion data",
        
        config = json_hyp.copy()
    )
    main(yaml_config, json_hyp)
    wandb.finish()
