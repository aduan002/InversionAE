import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_shape, linear_features, linear_num_hidden_layers,
                       lstm_input_shape, lstm_hidden_size, lstm_num_hidden_layers, lstm_dropout) -> None:
        super().__init__()

        out_features = linear_features
        self.linear_layers = nn.ModuleList()

        self.linear_layers.append(nn.Linear(in_features=input_shape[0], out_features=out_features))
        self.linear_layers.append(nn.BatchNorm1d(num_features=out_features))
        self.linear_layers.append(nn.ReLU())
        for i in range(1, linear_num_hidden_layers + 1):
            in_features = out_features
            out_features = out_features // 2

            self.linear_layers.append(nn.Linear(in_features=in_features, out_features=out_features))
            self.linear_layers.append(nn.BatchNorm1d(num_features=out_features))

            if i == linear_num_hidden_layers:
                self.linear_layers.append(nn.Tanh())
            else:
                self.linear_layers.append(nn.ReLU())

        self.lstm = nn.LSTM(input_size = lstm_input_shape[1], hidden_size = lstm_hidden_size, num_layers = lstm_num_hidden_layers, dropout = lstm_dropout)
        self.relu = nn.ReLU()
        self.lstm_L0 = nn.Linear(in_features=lstm_hidden_size, out_features=out_features)

    def forward(self, x_0, x_1):
        for layer in self.linear_layers:
            x_0 = layer(x_0)

        x_1, (h_t, c_t) = self.lstm(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.lstm_L0(x_1)
        # Just care about the output of the last LSTM in the sequence
        x_1 = x_1[:,-1,:]

        x = torch.add(x_0, x_1, alpha=1)
        return x

class Decoder(nn.Module):
    def __init__(self, output_shape, linear_features, num_hidden_layers) -> None:
        super().__init__()

        in_features = linear_features // 2**num_hidden_layers

        self.layers = nn.ModuleList()

        for _ in range(1, num_hidden_layers + 1):
            self.layers.append(nn.Linear(in_features=in_features, out_features=in_features * 2))
            self.layers.append(nn.BatchNorm1d(num_features=in_features * 2))
            self.layers.append(nn.ReLU())

            in_features *= 2
        self.layers.append(nn.Linear(in_features=in_features, out_features=output_shape[0]))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        #output = self.tanh(output) # NOTE: Right now, the input is between -15.9147 and 16.0816 by doing PCA and then Standard Scaling...
        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_out_shape, features=512, linear_num_hidden_layers = 3, 
                 lstm_input_shape = (None,2), lstm_hidden_size = 50, lstm_num_hidden_layers = 1, lstm_dropout = 0) -> None:
        super().__init__()

        self.encoder = Encoder(input_shape=in_out_shape, linear_features=features, linear_num_hidden_layers=linear_num_hidden_layers,
                               lstm_input_shape=lstm_input_shape, lstm_hidden_size=lstm_hidden_size, lstm_num_hidden_layers=lstm_num_hidden_layers, lstm_dropout=lstm_dropout)
        self.decoder = Decoder(output_shape=in_out_shape, linear_features=features, num_hidden_layers=linear_num_hidden_layers)

    def forward(self, x_0, x_1):
        x = self.encoder(x_0, x_1)
        x = self.decoder(x)
        return x