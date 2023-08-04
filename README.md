# Time Convolutional Autoencoder
This branch uses 3D Convolutional and LSTM layers to build an autoencoder on ERT and Weather data.

## Train
It trains the model with the given hyperparameters and data. It saves the scalers and reshapers used to preprocess the data as well as the weights with a given frequency.

The training script must be run with a given configuration and model hyperparemeters. An example of the command used to run the train script: `python train.py -c cfg/train.yaml -p cfg/hyp.json`. The `aux_coordinates` configuration parameter refers to a file containing the x, y, z coordinates for each tetrahedron like [spatial.txt](https://github.com/Aduan002/InversionAE/releases/download/v1.0.0/spatial.txt), and the `aux_weather` configuration parameter refers to a file containing rainfall and temperature data like [weather.csv](https://github.com/Aduan002/InversionAE/releases/download/v1.0.0/temporal.csv).

## Generate Tolerance Intervals
It creates upper bound tolerance intervals for each point in the ERT data with $\alpha$ confidence and $\beta$ coverage based on the percent error between the model prediction and the actual data. This is meant to be used on the training data to generate anomaly thresholds for each point.

The tolerance interval generation script must be run with a given configuration and model hyperparameters. An example of the command used to run the script: `python generate_tolerance_intervals.py -c cfg/tolerance.yaml`. The `aux_data` configuration parameter refers to the same `aux_weather` parameter from training, the `aux_coordinates` are not needed since the reshaper object has already been generated and saved.

## Predict 
It predicts on the given data using the model weights and tolerance intervals that were previously generated.

The prediction script must be run with a given configuration and model hyperparameters. An example of the command used to run the script: `python predict.py -c cfg/test.yaml -p cfg/hyp.json`. The `aux_data` configuration parameter refers to the same `aux_weather` parameter from training, the `aux_coordinates` are not needed since the reshaper object has already been generated and saved.
