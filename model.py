import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_shape, kernel_size = (3,3,1), stride = 1, padding = 0, num_hidden_layers = 2) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool3d(kernel_size=kernel_size))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_shape, kernel_size = (3,3,1), stride = 1, padding = 0, num_hidden_layers = 2) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_hidden_layers):
            self.layers.append(nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding))
            self.layers.append(nn.ReLU())

        self.output_shape = output_shape

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return nn.Upsample(size=self.output_shape[1:])(x) 

class AutoEncoder(nn.Module):
    def __init__(self, in_out_shape, kernel_size = (3,3,1), stride = 1, padding = 0, num_hidden_layers = 2) -> None:
        super().__init__()
        # in_out_shape should be channels, depth, height, width
        self.encoder = Encoder(input_shape=in_out_shape, kernel_size=kernel_size, stride=stride, padding=padding, num_hidden_layers=num_hidden_layers)
        self.decoder = Decoder(output_shape=in_out_shape, kernel_size=kernel_size, stride=stride, padding=padding, num_hidden_layers=num_hidden_layers)

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output