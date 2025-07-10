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

from utils.blenderproc import initialize_scene, add_plane, add_light, add_camera, set_camera_config, render_points, render, segmentation_handler

parser = argparse.ArgumentParser()

parser.add_argument("--smplx_pose_dir", type=str, default="rebuttal/fullbodymanip/test_data")
parser.add_argument("--category", type=str, default="clothesstand")
parser.add_argument("--text_annotation", type=str, default="Pull the clothesstand, and set it back down.")
parser.add_argument("--object_path", type=str, default="data/HOIs/cart/objects/00003.obj")
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--width", type=int, default=1200)
parser.add_argument("--height", type=int, default=800)

args = parser.parse_args()

COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64)

if __name__ == "__main__":
    initialize_scene()
  
    smplx_pose_pths = [sorted(glob(f"{args.smplx_pose_dir}/{args.category}/{args.text_annotation}/human/*.npz"))[args.idx]] # 1, 4

    for ii, smplx_pose_pth in enumerate(smplx_pose_pths):
        category = smplx_pose_pth.split("/")[-4]
        sample_idx = smplx_pose_pth.split("/")[-1].strip(".npz")

        # add smplx motion
        bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
        bpy.ops.object.smplx_add_animation(filepath=smplx_pose_pth)
        smplx = bpy.data.objects[f'SMPLX-neutral_{sample_idx}']
        smplx.delta_rotation_euler = (-np.pi/2,0,0)



        smplx_poses = np.load(smplx_pose_pth)
        frame_num = smplx_poses["poses"].shape[0]
        # largest_true_indices = smplx_poses["largest_true_indices"]

        scene = bpy.context.scene
        scene.render.resolution_x = args.width
        scene.render.resolution_y = args.height
        scene.render.resolution_percentage = 100

        # add camera
        bpy.ops.object.add(type="CAMERA")
        camera = bpy.context.object
        camera.name = "camera"
        cam_data = camera.data
        cam_data.name = "camera"
        cam_data.sensor_width = 10
        cam_data.lens = np.sqrt(args.width**2 + args.height**2) * cam_data.sensor_width / max(args.width, args.height)
        scene.camera = camera

        
        object_pose_path = smplx_pose_pth.replace("human", "estimated_object_pose_sequence").replace(".npz", "RT.pkl")
        object_gt_pose_path = smplx_pose_pth.replace("human", "object").replace(".npz", ".pkl")
        with open(object_pose_path, "rb") as handle: object_pose_data = pickle.load(handle)
        with open(object_gt_pose_path, "rb") as handle: object_gt_pose_data = pickle.load(handle)
        object_path = f"data/FullBodyManip/captured_objects/{category}_cleaned_simplified.obj"

        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]
        object_data = bpy.context.selected_objects[0].data
        # dict(
        #     obj_bps=manip_data["obj_bps"][0].cpu().numpy(), # 120 x 3
        #     obj_com_pos=manip_data["obj_com_pos"][0].cpu().numpy(), # 120 x 3072
        #     obj_rot_mat=manip_data["obj_rot_mat"][0].cpu().numpy(), # 120 x 3 x 3
        #     obj_scale=manip_data["obj_scale"][0].cpu().numpy(), # 120
        #     obj_trans=manip_data["obj_trans"][0].cpu().numpy(), # 120 x 3
        #     obj_bottom_rot_mat=manip_data["obj_bottom_rot_mat"][0].cpu().numpy(), # 120 x 3 x 3
        #     obj_bottom_scale=manip_data["obj_bottom_scale"][0].cpu().numpy(), # 120
        #     obj_bottom_trans=manip_data["obj_bottom_trans"][0].cpu().numpy(), # 120 x 3
        #     obj_name=obj_name
        # ),


        obj_Rs = object_pose_data["R"]
        obj_ts = object_pose_data["t"]
        # s = [object_gt_pose_data["obj_scale"][ii] for ii in largest_true_indices]
        s = object_gt_pose_data["obj_scale"]


        for frame_idx, (obj_R, obj_t) in enumerate(zip(obj_Rs, obj_ts)):
            
            object.rotation_euler = Matrix(obj_R).to_euler()
            object.location = Vector(obj_t.reshape((-1, )))
            object.scale = (s[frame_idx], s[frame_idx], s[frame_idx])

            object.keyframe_insert(data_path="rotation_euler", frame=frame_idx + 1)
            object.keyframe_insert(data_path="location", frame=frame_idx + 1)
            object.keyframe_insert(data_path="scale", frame=frame_idx + 1)



        bpy.ops.object.light_add(type="POINT", location=(0,0,0), rotation=(0,0,0))
        light = bpy.data.objects["Point"]
        light.data.energy = 1000
        
        # R = object_pose_data["obj_rot_mat"]
        # t = object_pose_data["obj_trans"]
        # s = object_pose_data["obj_scale"]

        # for frame_idx in range(1, frame_num + 1):
        #     rotation_euler = Matrix(COMPATIBILITY_MATRIX.T @ R[frame_idx - 1]).to_euler("XYZ")
        #     location = t[frame_idx - 1].reshape((3,)) @ COMPATIBILITY_MATRIX
        #     # rotation_euler[0] = np.pi + rotation_euler[0]
            
        #     object.rotation_euler = rotation_euler
        #     object.location = location
        #     object.scale = (s[frame_idx - 1], s[frame_idx - 1], s[frame_idx - 1])

        #     object.keyframe_insert(data_path="rotation_euler", frame=frame_idx + 1)
        #     object.keyframe_insert(data_path="location", frame=frame_idx + 1)
        #     object.keyframe_insert(data_path="scale", frame=frame_idx + 1)
