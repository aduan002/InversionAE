import torch
from torch import nn
import math

def find_output_shape(input_shape, kernel_size, padding, stride, dilation = 1):
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        
        if padding == "valid":
            padding = (0,0,0)
        elif padding == "same": # NOTE: dilation * (kernel_size - 1) / 2 will give a padding such that shape of input equals shape of output... stride must be 1 for 'same'.
            padding = (
                dilation[0] * (kernel_size[0] - 1) / 2,
                dilation[1] * (kernel_size[1] - 1) / 2,
                dilation[2] * (kernel_size[2] - 1) / 2,
            )
        elif isinstance(padding, int):
            padding = (padding, padding, padding)

        # NOTE: See https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        channels, d_in, h_in, w_in = input_shape
        d_out = math.floor((d_in + 2*padding[0] - dilation[0]*(kernel_size[0] - 1) - 1) / stride[0] + 1)
        h_out = math.floor((h_in + 2*padding[1] - dilation[1]*(kernel_size[1] - 1) - 1) / stride[1] + 1)
        w_out = math.floor((w_in + 2*padding[2] - dilation[2]*(kernel_size[2] - 1) - 1) / stride[2] + 1)

        return channels, d_out, h_out, w_out


class Encoder(nn.Module):
    def __init__(self, input_shape, kernel_size, stride, padding, conv_num_hidden_layers,
                       lstm_input_shape, lstm_hidden_size, lstm_num_hidden_layers, lstm_dropout) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        d_out, h_out, w_out = None, None, None
        for _ in range(conv_num_hidden_layers):
            self.layers.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding))
            self.layers.append(nn.ReLU())

            # NOTE: The channels are kept the same throughout...
            channels, d_out, h_out, w_out = find_output_shape(input_shape, kernel_size, padding, stride)
            input_shape = (channels, d_out, h_out, w_out)

        self.layers.append(nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding))
        channels, d_out, h_out, w_out = find_output_shape(input_shape, kernel_size, padding, stride)
        input_shape = (channels, d_out, h_out, w_out)


        self.lstm = nn.LSTM(input_size = lstm_input_shape[1], hidden_size = lstm_hidden_size, num_layers = lstm_num_hidden_layers, dropout = lstm_dropout)
        self.relu = nn.ReLU()
        self.lstm_L0 = nn.Linear(in_features=lstm_hidden_size, out_features=d_out*h_out*w_out)


    def forward(self, x_0, x_1):
        for layer in self.layers:
            x_0 = layer(x_0)
        
        x_1, (h_t, c_t) = self.lstm(x_1)
        x_1 = self.relu(x_1)
        x_1 = x_1[:,-1,:]
        x_1 = self.lstm_L0(x_1)
        # Just care about the output of the last LSTM in the sequence
        
        x_1 = torch.reshape(x_1, x_0.shape)

        x = torch.add(x_0, x_1, alpha=1)
        return x

class Decoder(nn.Module):
    def __init__(self, output_shape, kernel_size, stride, padding, num_hidden_layers, upsample_mode) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_hidden_layers):
            self.layers.append(nn.ConvTranspose3d(in_channels=1, out_channels=1, kernel_size=kernel_size, stride=stride, padding=padding))
            self.layers.append(nn.ReLU())
        self.upsample = nn.Upsample(size = output_shape[1:], mode = upsample_mode)

        self.output_shape = output_shape

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.upsample(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_out_shape, kernel_size = (3,3,1), stride = 1, padding = 0, conv_num_hidden_layers = 2, upsample_mode = "nearest",
                       lstm_input_shape = (None, 2), lstm_hidden_size = 50, lstm_num_hidden_layers = 1, lstm_dropout = 0) -> None:
        super().__init__()
        # in_out_shape should be channels, depth, height, width
        self.encoder = Encoder(input_shape=in_out_shape, kernel_size=kernel_size, stride=stride, padding=padding, conv_num_hidden_layers=conv_num_hidden_layers,
                               lstm_input_shape=lstm_input_shape, lstm_hidden_size=lstm_hidden_size, lstm_num_hidden_layers=lstm_num_hidden_layers, lstm_dropout=lstm_dropout)
        self.decoder = Decoder(output_shape=in_out_shape, kernel_size=kernel_size, stride=stride, padding=padding, num_hidden_layers=conv_num_hidden_layers, upsample_mode=upsample_mode)

    def forward(self, x_0, x_1):
        x = self.encoder(x_0, x_1)
        x = self.decoder(x)
        return x