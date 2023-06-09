import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        self.L1 = nn.Linear(in_features=input_shape[0], out_features=512)
        self.B1 = nn.BatchNorm1d(num_features=512)
        self.L2 = nn.Linear(in_features=512, out_features=256)
        self.B2 = nn.BatchNorm1d(num_features=256)
        self.L3 = nn.Linear(in_features=256, out_features=128)
        self.B3 = nn.BatchNorm1d(num_features=128)
        self.L4 = nn.Linear(in_features=128, out_features=64)
        self.B4 = nn.BatchNorm1d(num_features=64)

    def forward(self, x):
        output = self.L1(x)
        output = self.B1(output)
        output = self.relu(output)
        output = self.L2(output)
        output = self.B2(output)
        output = self.relu(output)
        output = self.L3(output)
        output = self.B3(output)
        output = self.relu(output)
        output = self.L4(output)
        output = self.B4(output)
        output = self.tanh(output)
        return output

class Decoder(nn.Module):
    def __init__(self, output_shape) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.L1 = nn.Linear(in_features=64, out_features=128)
        self.B1 = nn.BatchNorm1d(num_features=128)
        self.L2 = nn.Linear(in_features=128, out_features=256)
        self.B2 = nn.BatchNorm1d(num_features=256)
        self.L3 = nn.Linear(in_features=256, out_features=512)
        self.B3 = nn.BatchNorm1d(num_features=512)
        self.L4 = nn.Linear(in_features=512, out_features=output_shape[0])

    def forward(self, x):
        output = self.L1(x)
        output = self.B1(output)
        output = self.relu(output)
        output = self.L2(output)
        output = self.B2(output)
        output = self.relu(output)
        output = self.L3(output)
        output = self.B3(output)
        output = self.relu(output)
        output = self.L4(output)
        #output = self.tanh(output) # NOTE: Right now, the input is between -15.9147 and 16.0816 by doing PCA and then Standard Scaling...
        return output

class AutoEncoder(nn.Module):
    def __init__(self, in_out_shape) -> None:
        super().__init__()
        self.encoder = Encoder(input_shape=in_out_shape)
        self.decoder = Decoder(output_shape=in_out_shape)

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output