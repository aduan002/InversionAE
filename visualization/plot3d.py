import os 
import pandas as pd
import numpy as np
import argparse
import yaml 
from tqdm import tqdm
import tempfile

from matplotlib.colors import LinearSegmentedColormap
import pyvista as pv

def obj_data_to_mesh3d(file_path):
    vertices = []
    faces = []
    with open(file_path, "r") as file:
        for line in file:
            slist = line.split()
            if slist:
                if slist[0] == 'v':
                    vertex = np.array(slist[1:], dtype=float)
                    vertices.append(vertex)
                elif slist[0] == 'f':
                    face = []
                    for k in range(1, len(slist)):
                        face.append([int(s) for s in slist[k].replace('//','/').split('/')])
                    if len(face) > 3: # triangulate the n-polyonal face, n>3
                        faces.extend([[face[0][0]-1, face[k][0]-1, face[k+1][0]-1] for k in range(1, len(face)-1)])
                    else:
                        faces.append([face[j][0]-1 for j in range(len(face))])
                else: pass


    return np.array(vertices), np.array(faces)

def main(config):
    SCALE_FACTOR = 1.5

    input_path  = config["input_path"]
    save_dir = config["save_dir"]
    show_plot = config["show_plot"]
    save_image = config["save_image"]
    color_scheme = config["color_scheme"]

    colors = []
    for color in color_scheme:
        r,g,b = color.split(",")
        colors.append((float(r),float(g),float(b)))


    input_dir = None
    if not os.path.isdir(input_path):
        input_dir, filename = os.path.split(input_path)
        filenames = [filename]
    else:
        input_dir = input_path
        filenames = os.listdir(input_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cmap = LinearSegmentedColormap.from_list(name="Plot3D Custom Color Scheme", colors=colors)

    for filename in filenames:
        vertices, faces = obj_data_to_mesh3d(os.path.join(input_dir, filename))
        xyz = vertices[:, :3]
        w = vertices[:, 3]

        point_cloud = pv.PolyData(xyz)
        point_cloud["Error"] = w

        # There is an issue where if you show the window and exit, the plot closes even though auto_close=False
        if show_plot is True:
            plotter = pv.Plotter(off_screen=False, title=filename)
            plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=7, cmap=cmap, clim=[0.0,100.0])
            plotter.show_axes()
            plotter.show()

        if save_image is True:
            plotter = pv.Plotter(off_screen=True, title=filename)
            plotter.add_mesh(point_cloud, render_points_as_spheres=True, point_size=7, cmap=cmap, clim=[0.0,100.0])
            plotter.show_axes()
            plotter.show(auto_close=False)
            plotter.camera.zoom(SCALE_FACTOR)

            # NOTE: Screenshots don't save with the zoom, so this is a hacky workaround to force a zoom on the screenshot.
            temp_file = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
            plotter.save_graphic(temp_file.name)
            temp_file.close()
            os.remove(temp_file.name)

            plotter.screenshot(os.path.join(save_dir, filename.split(".")[0] + ".png"), transparent_background=True)

            """ # NOTE: Some samples code to do a full orbit around the x-axis.
            plotter.show(auto_close=False)
            path = plotter.generate_orbital_path(n_points=300, shift=point_cloud.length)
            plotter.open_movie("pyvista.mp4")
            plotter.orbit_on_path(path, write_frames=True, progress_bar=True)
            plotter.close()
            """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "Point Cloud Plotter",
         description = "Plot a Point Cloud of the prediction errors"
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

    