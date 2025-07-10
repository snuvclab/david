import blenderproc as bproc
import bpy
import bmesh
from bpy.app.handlers import persistent

import numpy as np
import cv2

from glob import glob
from mathutils import Matrix, Vector
import argparse
import pickle
import sys
import os
from PIL import Image
import bpy
from bpy_extras.object_utils import world_to_camera_view
from tqdm import tqdm

sys.path.append(os.getcwd())

from constants.config import WIDTH, HEIGHT
from utils.blenderproc import initialize_scene, add_plane, add_light, add_camera, set_camera_config, render_points, render, segmentation_handler
from utils.dataset import category2object

COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)


def visualize_david_output(
    dataset,
    category,
    human_motion_dir,
    object_motion_dir,
    idx,
):
    initialize_scene()
   
    smplx_pose_pths = [sorted(glob(f"{human_motion_dir}/{dataset}/{category}/*/*/*.npz"))[idx]]

    print(smplx_pose_pths)
    for ii, smplx_pose_pth in enumerate(smplx_pose_pths):
        object_motion_pth = smplx_pose_pth.replace(human_motion_dir, object_motion_dir).replace(".npz", ".pkl")
        # add smplx motion
        bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
        bpy.ops.object.smplx_add_animation(filepath=smplx_pose_pth)
        smplx = bpy.data.objects['SMPLX-neutral_sample00_rep00_smpl_params']
        smplx.delta_rotation_euler = (-np.pi/2, 0, 0)

        smplx_poses = np.load(smplx_pose_pth)
        frame_num = smplx_poses["poses"].shape[0]

        scene = bpy.context.scene
        scene.render.resolution_x = WIDTH
        scene.render.resolution_y = HEIGHT
        scene.render.resolution_percentage = 100

        # add camera
        bpy.ops.object.add(type="CAMERA")
        camera = bpy.context.object
        camera.name = "camera"
        cam_data = camera.data
        cam_data.name = "camera"
        cam_data.sensor_width = 10
        cam_data.lens = np.sqrt(WIDTH**2 + HEIGHT**2) * cam_data.sensor_width / max(WIDTH, HEIGHT)
        scene.camera = camera

        object_path = category2object(dataset, category)
    
        bpy.ops.object.light_add(type="POINT", location=(0,0,0), rotation=(0,0,0))
        light = bpy.data.objects["Point"]
        light.data.energy = 1000

        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]
        object_data = bpy.context.selected_objects[0].data

        with open(object_motion_pth, "rb") as handle:
            obj_motion = pickle.load(handle)
        obj_Rs = obj_motion["R"]
        obj_ts = obj_motion["t"]
        average_scale = obj_motion["average_scale"]

        for frame_idx, (obj_R, obj_t) in enumerate(zip(obj_Rs, obj_ts)):
            object.rotation_euler = Matrix(obj_R).to_euler()
            object.location = Vector(obj_t.reshape((-1, )))
            object.scale = (average_scale, average_scale, average_scale)

            object.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            object.keyframe_insert(data_path="location", frame=frame_idx)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # visualize configuration
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")

    parser.add_argument("--human_motion_dir", type=str, default="results/inference/human_motion")
    parser.add_argument("--object_motion_dir", type=str, default="results/inference/object_motion")

    parser.add_argument("--idx", type=int, default=0)

    args = parser.parse_args()

    visualize_david_output(
        dataset=args.dataset,
        category=args.category,
        human_motion_dir=args.human_motion_dir,
        object_motion_dir=args.object_motion_dir,
        idx=args.idx,
    )