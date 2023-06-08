import torch
from torch.utils.data import DataLoader
import wandb
import argparse
import yaml
from tqdm import tqdm
import os

from model import AutoEncoder
from dataset import InversionDataset
from scaler import CustomMinMaxScaler

def train_loop(dataloader, model, loss_fn, optimizer):
    # Set the model to training mode - important for batch normalization and dropout layers
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    for batch, tensor in enumerate(tqdm(dataloader)):
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

def val_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    model.eval()
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for batch, tensor in enumerate(tqdm(dataloader)):
            pred = model(tensor)
            loss = loss_fn(pred, tensor)

            val_loss += loss.item()

    val_loss /= num_batches
    return val_loss

def save_model(model, save_dir, save_name):
    torch.save(model.state_dict(), os.path.join(save_dir, save_name))

def main(config):    
    train_dir = config["data"]["train"]
    val_dir = config["data"]["val"]

    batch_size = config["hp"]["batch_size"]
    epochs = config["hp"]["epochs"]
    learning_rate = config["hp"]["learning_rate"]
    weight_decay = config["hp"]["weight_decay"]

    save_dir = config["model"]["save_dir"]
    save_freq = config["model"]["save_freq"]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    scaler = CustomMinMaxScaler()
    scaler.fit(train_dir)

    train_data = InversionDataset(train_dir, scaler=scaler)
    val_data = InversionDataset(val_dir, scaler=scaler)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)

    model = AutoEncoder(in_out_shape = train_data.__getitem__(0).shape)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr = learning_rate,
                                weight_decay = weight_decay)
    
    for e in range(1, epochs + 1):
        print("Epoch:", e)
        
        print("Training:")
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        
        print("Validation:")
        val_loss = val_loop(val_dataloader, model, loss_fn) 

        wandb.log({"train_mse_loss": train_loss, "val_mse_loss": val_loss})

        if e % save_freq == 0:
            save_name = str(e).zfill(len(str(epochs))) + ".pt"
            save_model(model, save_dir, save_name)

    scaler.save("scalers", "min_max_scaler.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Linear AutoEncoder",
         description = "Training an AutoEncoder using Linear Layers"
    )
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as file:
        try:
            yaml_config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    wandb.init(
        #mode="disabled",

        project="Linear AutoEncoder",
        name="Run 1",
        notes="Training Linear AutoEncoder on Inversion data",
        
        config = yaml_config.copy()
    )
    main(yaml_config)
    wandb.finish()
