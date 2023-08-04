# Auxiliary Code
This branch contains code that is not deep learning specific but still needed for the system. This includes filtering and splitting the data, generating anomalies for testing, and visualizing the data.

## Data Split
The `train_test_split.py` script splits the data into train, validation, and test sets that have a distribution similar to the distribution.
The `filter.py` script uses [spatial.txt](https://github.com/Aduan002/InversionAE/tree/AuxiliaryCode) to choose which rows of the data to use.

## Generate Anomalies
The `generate_anomalies.py` script increases the conductivity values in a radius R around a point P by a factor N. It uses [node](https://github.com/Aduan002/InversionAE/tree/AuxiliaryCode), [element](https://github.com/Aduan002/InversionAE/tree/AuxiliaryCode), and [translation](https://github.com/Aduan002/InversionAE/tree/AuxiliaryCode) files. The `node` file contains each point and its x, y, and z position. The `element` file contains the tetrahedron index, the four points that make up the tetrahedron and the zone the tetrahedron belongs to. The `translation` file contains x, y, and z translations on the x, y, and z values in the `node` file.

The `generate_filtered_anomalies.py` script is the same as the `generate_anomalies.py` script but the elements and zones are filtered before being used to create anomalies on already filtered data.

## Visualization
The `convert_to_obj.py` script takes the result of the prediction and converts it to an obj file format where each row is a vertex with an x, y, z, and w values where w is the percent error after prediction. It YAML needs a configuration file as input where the `spatial_data` parameter refers to the `spatial.txt` file used when filtering the data.

The `cluster_regression.py` script is similar to the `convert_to_obj.py` script with one change that applies KNN style regression where the percent error of points is adjusted once to be the average percent error of its N neighbors (before adjustment). It needs a YAML configuration file as input where the `spatial_data` parameter refers to the `spatial.txt` file used when filtering the data.

The `plotk3d.py` script takes the obj files as input and converts them into an html 3D Visualization of the data using [K3D](https://github.com/K3D-tools/K3D-jupyter). It needs a YAML configuration file as input where the `color_scheme` parameter needs a list of colors in hex format.

The `plot3d.py` script is similar to the `plotk3d.py` script that takes the obj files as input but outputs a 3D visualization as a python app window and saves an isometric view of the visualization. The 3D visualization was done using [pyvista](https://github.com/pyvista/pyvista). This script was implemented before the `plotk3d.py` script, so it can be considered **deprecated**.
