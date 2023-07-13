import os 
import numpy as np
import argparse
import yaml 
from tqdm import tqdm

import k3d

import io
import zlib
import base64

def get_html(plot, title, compression_level=9, voxel_chunks=[], additional_js_code=""):
    dir_path = os.path.dirname(os.path.realpath(k3d.__file__))

    data = plot.get_binary_snapshot(compression_level, voxel_chunks)

    if plot.snapshot_type == 'full':
        f = io.open(
            os.path.join(dir_path, "static", "snapshot_standalone.txt"),
            mode="r",
            encoding="utf-8",
        )
        template = f.read()
        template = template.replace("K3D snapshot viewer - [TIMESTAMP]", title)
        f.close()

        f = io.open(
            os.path.join(dir_path, "static", "standalone.js"),
            mode="r",
            encoding="utf-8",
        )
        template = template.replace(
            "[K3D_SOURCE]",
            base64.b64encode(
                zlib.compress(f.read().encode(), compression_level)
            ).decode("utf-8"),
        )
        f.close()

        f = io.open(
            os.path.join(dir_path, "static", "require.js"),
            mode="r",
            encoding="utf-8",
        )
        template = template.replace("[REQUIRE_JS]", f.read())
        f.close()

        f = io.open(
            os.path.join(dir_path, "static", "fflate.js"),
            mode="r",
            encoding="utf-8",
        )
        template = template.replace("[FFLATE_JS]", f.read())
        f.close()
    else:
        if plot.snapshot_type == 'online':
            template_file = 'snapshot_online.txt'
        elif plot.snapshot_type == 'inline':
            template_file = 'snapshot_inline.txt'
        else:
            raise Exception('Unknown snapshot_type')

        f = io.open(
            os.path.join(dir_path, "static", template_file),
            mode="r",
            encoding="utf-8",
        )
        template = f.read()
        f.close()

        template = template.replace("[VERSION]", plot._view_module_version)
        template = template.replace("[HEIGHT]", str(plot.height))
        template = template.replace("[ID]", str(id(plot)))

    template = template.replace("[DATA]", base64.b64encode(data).decode("utf-8"))
    template = template.replace("[ADDITIONAL]", additional_js_code)

    return template

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

def get_colors(color_scheme, spacing, w):
    colors = []
    norm_w = w / 100 # Percent error is a percent, we want decimals.
    for v in norm_w:
        color = None
        for idx in range(len(spacing)):
            if v < spacing[idx]:
                color = color_scheme[idx]
                break
        if color is None:
            color = color_scheme[len(color_scheme) - 1]

        colors.append(color)
    return colors


def main(config):
    OFFSET = 1e-6

    input_path  = config["input_path"]
    save_dir = config["save_dir"]
    color_scheme = config["color_scheme"]

    spacing = []
    space = (1 - OFFSET) / (len(color_scheme) - 1)
    for idx in range(1, len(color_scheme)):
        spacing.append(space * idx + OFFSET)
    spacing.insert(0, OFFSET)


    input_dir = None
    if not os.path.isdir(input_path):
        input_dir, filename = os.path.split(input_path)
        filenames = [filename]
    else:
        input_dir = input_path
        filenames = os.listdir(input_path)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    for filename in tqdm(filenames):
        vertices, faces = obj_data_to_mesh3d(os.path.join(input_dir, filename))
        xyz = vertices[:, :3]
        w = vertices[:, 3]
        colors = get_colors(color_scheme=color_scheme, spacing=spacing, w=w)

        plot = k3d.plot(grid_visible=False, menu_visibility=False, background_color=0x6A6A6A, camera_mode="orbit")
        points = k3d.points(xyz.astype(np.float32), point_size=0.3, colors=colors)  # The points only support up to float32
        plot += points

        title = filename.split(".")[0]
        save_path = os.path.join(save_dir, title + ".html")

        with open(save_path, "w") as file:
            file.write(get_html(plot, title))




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

    