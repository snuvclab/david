import blenderproc as bproc
import bpy

import os
import sys

sys.path.append(os.getcwd())

import json
import pickle
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np

from utils.transformations import deg2rad
from utils.blenderproc import initialize_scene, add_plane, add_light, add_camera, set_camera_config, render
from utils.dataset import category2object
from constants.config import CATEGORY2RENDERCONFIG
from constants.metadata import DEFAULT_SEED

def render_objects(
    dataset,
    category,
    object_image_save_dir,
    camera_dir,
    default_resolution,
    default_elevation,
    default_azimuth,
    default_view_num,
    default_z_transl,
    default_radius,
    skip_done,
    verbose,
):
    # blender settings
    plane = add_plane()
    add_light()
    bproc.renderer.enable_depth_output(activate_antialiasing=True)

    if CATEGORY2RENDERCONFIG.get(category, None) is None:
        camera_config = CATEGORY2RENDERCONFIG["default"]
    else:
        camera_config = CATEGORY2RENDERCONFIG[category]

    focal_length = camera_config["focal_length"]
    scale = camera_config["scale"]
    resolution = default_resolution  ## change here if wanted
    camera = add_camera(default_resolution, focal_length, "CAMERA")

    os.makedirs(f"{object_image_save_dir}/{dataset}/{category}", exist_ok=True)
    os.makedirs(f"{camera_dir}/{dataset}/{category}", exist_ok=True)

    # prepare asset
    object_path = category2object(dataset, category)
    if object_path.endswith(".obj"): bpy.ops.import_scene.obj(filepath=object_path)
    elif object_path.endswith(".ply"): bpy.ops.import_mesh.ply(filepath=object_path)
    object = bpy.context.selected_objects[0]

    if CATEGORY2RENDERCONFIG[category].get("initial_scale", None) is not None:
        s = CATEGORY2RENDERCONFIG[category]["initial_scale"]
        object.scale = (s, s, s)
    
    if CATEGORY2RENDERCONFIG[category].get("initial_rotation", None) is not None:
        rotation_euler = np.array(CATEGORY2RENDERCONFIG[category]["initial_rotation"]) / 180 * np.pi
        object.rotation_euler = rotation_euler

    # setting camera configuration
    elevation = deg2rad(camera_config.get("elevation", default_elevation))
    azimuth = deg2rad(camera_config.get("azimuth", default_azimuth))
    view_num = camera_config.get("view_num", default_view_num)
    z_transl = camera_config.get("z_transl", default_z_transl)
    radius = camera_config.get("radius", default_radius)

    cameras = [
        dict(
            location=(
                radius * np.cos(elevation) * np.cos(azimuth + (2 * np.pi / view_num) * view_idx),
                radius * np.cos(elevation) * np.sin(azimuth + (2 * np.pi / view_num) * view_idx),
                radius * np.sin(elevation) + z_transl,
            ),
            rotation=(np.pi / 2 - elevation, 0, np.pi / 2 + azimuth + (2 * np.pi / view_num) * view_idx),
        )
        for view_idx in range(view_num)
    ]

    # run for rendering cameras
    for idx, render_camera in enumerate(cameras):
        set_camera_config(scale=scale, location=render_camera["location"], rotation=render_camera["rotation"])

        # save camera
        scene = bpy.context.scene
        camera = scene.camera
        camera_matrix_world = camera.matrix_world

        # render object
        object_render_save_pth = f"{object_image_save_dir}/{dataset}/{category}/view:{idx:05d}.png"
        if not os.path.exists(object_render_save_pth) or not skip_done:
            render(object_render_save_pth, hide_objects=[plane], use_gpu=True, desc="multiview object rendering")
        elif verbose:
            print(f"{object_render_save_pth} already processed!!")
        
        R = np.array(camera_matrix_world)[:3, :3]  # 3 x 3
        t = np.array(camera_matrix_world)[:3, 3]  # 3 x 1

        camera_save_pth = f"{camera_dir}/{dataset}/{category}/view:{idx:05d}.pickle"
        if not os.path.exists(camera_save_pth) or not skip_done:
            with open(camera_save_pth, "wb") as handle:
                pickle.dump(
                    dict(
                        R=R,
                        t=t,

                        obj_R=np.array(object.rotation_euler.to_matrix()).reshape((3, 3)),  # 3 x 3
                        obj_euler=np.array(object.rotation_euler).reshape((3, 1)),  # 3 x 1
                        obj_location=np.array(object.location).reshape((3, 1)),  # 3 x 1
                        obj_t=np.array(object.location).reshape((3, 1)),  # 3 x 1
                        obj_scale=np.array(object.scale).reshape((3, 1)),  # 3 x 1

                        focal_length=focal_length,
                        resolution=resolution,
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
        elif verbose:
            print(f"{camera_save_pth} already processed!!")


def render_scene(args):
    ## initialize blenderproc scene
    initialize_scene()
    render_objects(
        dataset=args.dataset,
        category=args.category,
        object_image_save_dir=args.object_image_save_dir,
        camera_dir=args.camera_dir,
        default_resolution=(args.default_resolution_x, args.default_resolution_y),
        default_elevation=args.default_elevation,
        default_azimuth=args.default_azimuth,
        default_view_num=args.default_view_num,
        default_z_transl=args.default_z_transl,
        default_radius=args.default_radius,
        skip_done=args.skip_done,
        verbose=args.verbose
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## object information
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")

    ## save directories
    parser.add_argument("--object_image_save_dir", type=str, default="results/generation/asset_renders")
    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")

    ## input directories
    parser.add_argument("--default_resolution_x", type=int, default=1200)
    parser.add_argument("--default_resolution_y", type=int, default=800)

    parser.add_argument("--default_elevation", type=float, default=7.5)
    parser.add_argument("--default_azimuth", type=float, default=45)
    parser.add_argument("--default_view_num", type=int, default=8)
    parser.add_argument("--default_z_transl", type=float, default=0.2)
    parser.add_argument("--default_radius", type=float, default=2.0)

    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    render_scene(args)
