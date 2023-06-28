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
from transformations.scaler import PartialStandardScaler, FullStandardScaler
from transformations.pca import CustomPCA

def train_loop(dataloader, model, loss_fn, optimizer, device):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    for batch, (tensor, aux_tensor, _) in enumerate(tqdm(dataloader)):
        tensor = tensor.to(device)
        aux_tensor = aux_tensor.to(device)
        # Compute prediction and loss
        pred = model(tensor, aux_tensor)
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
        for batch, (tensor, aux_tensor, _) in enumerate(tqdm(dataloader)):
            tensor = tensor.to(device)
            aux_tensor = aux_tensor.to(device)
            pred = model(tensor, aux_tensor)
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
    aux_filepath = config["data"]["aux_data"]

    batch_size = hyp["batch_size"]
    epochs = hyp["epochs"]
    learning_rate = hyp["learning_rate"]
    weight_decay = hyp["weight_decay"]
    patience = hyp["patience"]

    linear_num_hidden_layers = hyp["linear_num_hidden_layers"]
    lstm_hidden_size = hyp["lstm_hidden_size"]
    lstm_num_hidden_layers = hyp["lstm_num_hidden_layers"]
    lstm_dropout = hyp["lstm_dropout"]
    lstm_steps = hyp["lstm_steps"]

    save_dir = config["model"]["save_dir"]
    save_freq = config["model"]["save_freq"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # PCA
    pca = CustomPCA()
    pca.fit(train_dir)
    pca.save("PCAs", "pca.pickle")

    # Scale
    scaler = PartialStandardScaler()
    scaler.fit(train_dir, pca)
    scaler.save("Scalers", "partial_standard_scaler.pickle")

    aux_scaler = FullStandardScaler()

    train_data = InversionDataset(train_dir, aux_filepath, scaler=scaler, pca=pca, aux_scaler=aux_scaler, timesteps=lstm_steps)
    val_data = InversionDataset(val_dir, aux_filepath, scaler=scaler, pca=pca, aux_scaler=aux_scaler, timesteps=lstm_steps)

    aux_scaler.save("Scalers", "full_standard_scaler.pickle")

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    model = AutoEncoder(in_out_shape = train_data.__getitem__(0)[0].shape, 
                        linear_num_hidden_layers=linear_num_hidden_layers, 
                        lstm_input_shape=train_data.__getitem__(0)[1].shape,
                        lstm_hidden_size=lstm_hidden_size,
                        lstm_num_hidden_layers=lstm_num_hidden_layers,
                        lstm_dropout=lstm_dropout)
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
         prog = "Time Linear AutoEncoder",
         description = "Training an AutoEncoder using Linear and LSTM Layers"
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
        mode="disabled",

        project="Time Linear AutoEncoder",
        name="Run 1",
        notes="Training Linear AutoEncoder on Inversion data and Weather data",
        
        config = json_hyp.copy()
    )
    main(yaml_config, json_hyp)
    wandb.finish()
