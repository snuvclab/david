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


def visualize_4d_hoi_sample(
    dataset,
    category,
    hmr_dir,
    camera_dir,
    camera_motion_dir,
    scale_dir,
    pnp_dir,
    idx,
):
    initialize_scene()
   
    smplx_pose_pths = [sorted(glob(f"{hmr_dir}/{dataset}/{category}/*/*/*.npz"))[idx]]

    print(smplx_pose_pths)
    for ii, smplx_pose_pth in enumerate(smplx_pose_pths):
        # add smplx motion
        bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
        bpy.ops.object.smplx_add_animation(filepath=smplx_pose_pth)

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

        category = smplx_pose_pth.split("/")[-4]
        camera_motion_path = smplx_pose_pth.replace(hmr_dir, camera_motion_dir).replace(".npz", ".pkl")
        scale_path = smplx_pose_pth.replace(hmr_dir, scale_dir).replace(".npz", ".pkl")
        pnp_path = smplx_pose_pth.replace(hmr_dir, pnp_dir).replace(".npz", ".pkl")
        
        category = smplx_pose_pth.split("/")[-4]
        object_path = category2object(dataset, category)

        ## ADD CAMERA VIS
        camera_object_path = f"camera_obj/model.obj"
        bpy.ops.import_scene.obj(filepath=camera_object_path)
        camera_object = bpy.context.selected_objects[0]
        camera_object_data = bpy.context.selected_objects[0].data

        bpy.ops.object.light_add(type="POINT", location=(0,0,0), rotation=(0,0,0))
        light = bpy.data.objects["Point"]
        light.data.energy = 1000

        with open(camera_motion_path, 'rb') as handle:
            camera_data = pickle.load(handle)
        with open(pnp_path, 'rb') as handle:
            pnp_results = pickle.load(handle)

        R = camera_data['R']
        t = camera_data['t']
        
        for frame_idx in range(1, frame_num + 1):
            rotation_euler = Matrix(COMPATIBILITY_MATRIX.T @ R[frame_idx - 1]).to_euler("XYZ")
            location = t[frame_idx - 1].reshape((3,)) @ COMPATIBILITY_MATRIX
            rotation_euler[0] = np.pi + rotation_euler[0]

            camera.rotation_euler = rotation_euler
            camera.location = location.reshape((-1,))

            camera.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            camera.keyframe_insert(data_path="location", frame=frame_idx)

            ## ADD CAMERA VIS
            camera_rotation_euler = rotation_euler
            camera_rotation_euler[0] += np.pi / 2
            # camera_object.rotation_euler = camera_rotation_euler 
            # camera_object.location = location.reshape((-1,))
            # camera_object.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            # camera_object.keyframe_insert(data_path="location", frame=frame_idx)
            camera_object.rotation_euler = (np.pi / 2, 0.0, 0.0)
            camera_object.location = (0.0, 0.0, 0.0)
            camera_object.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            camera_object.keyframe_insert(data_path="location", frame=frame_idx)

            light.rotation_euler = rotation_euler
            light.location = location.reshape((-1,))

            light.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            light.keyframe_insert(data_path="location", frame=frame_idx)

            if frame_idx == 1:
                initial_camera_rotation_euler = np.array(camera.rotation_euler.to_matrix())
                initial_camera_translation = np.array(camera.location).reshape((3, 1))

        render_camera_pth = "/".join(smplx_pose_pth.replace(hmr_dir, camera_dir).split("/")[:-2]) + ".pickle"
        with open(render_camera_pth, "rb") as handle:
            render_camera_data = pickle.load(handle)
        
        render_camera_R = np.array(render_camera_data["R"])
        render_camera_t = np.array(render_camera_data["t"]).reshape((3, 1))


        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]
        object_data = bpy.context.selected_objects[0].data

        initial_scale = 1.0
        if render_camera_data.get("obj_euler", None) is not None:
            object.rotation_euler = render_camera_data["obj_euler"].reshape((-1, ))
        if render_camera_data.get("obj_location", None) is not None:
            object.location = render_camera_data["obj_location"].reshape((-1, ))
        if render_camera_data.get("obj_scale", None) is not None:
            object.scale = render_camera_data["obj_scale"].reshape((-1, ))
            initial_scale = float(render_camera_data["obj_scale"][0])

        ## ADDED AFTER PERTURBATION SETTING
        object_rotation = np.array(object.rotation_euler.to_matrix())
        object_translation = np.array(object.location).reshape((3, 1))

        tranform_R = initial_camera_rotation_euler @ np.linalg.inv(render_camera_R)
        transformed_t = tranform_R @ (object_translation - render_camera_t) + initial_camera_translation
        
        transformed_rotation_euler = Matrix(tranform_R @ object_rotation).to_euler("XYZ")
        transformed_location = transformed_t.reshape((-1, ))

        object.rotation_euler = transformed_rotation_euler
        object.location = transformed_location

        initial_distance_camcenter2object = np.sqrt(np.sum(initial_camera_translation - transformed_location) ** 2)
        directional_vector = transformed_location.reshape((3, 1)) - initial_camera_translation.reshape((3, 1))

        bpy.context.view_layer.update()

        with open(scale_path, "rb") as handle:
            scale_data = pickle.load(handle)

        scale = scale_data["scale"]

        print(initial_scale)
        print(scale)
        # scale = scale_data["scale_mean"]

        for idx, pnp_result in enumerate(pnp_results):
            pnp_cam_R = pnp_result["inferred_cam_R"] # 3 x 3
            pnp_cam_t = pnp_result["inferred_cam_t"] # 3 x 1
            
            # real_camera_R = pnp_result["real_camera_R"] # 3 x 3
            # real_camera_t = pnp_result["real_camera_t"] # 3 x 1

            real_camera_R = np.eye(3).reshape((3, 3))
            real_camera_t = np.zeros((3, 1)).reshape((3, 1))

            tranform_R = -real_camera_R @ np.linalg.inv(pnp_cam_R)
            transformed_t = tranform_R @ (transformed_location.reshape((3, 1)) - pnp_cam_t) + real_camera_t

            directional_vector = transformed_t.reshape((3, 1)) - real_camera_t.reshape((3, 1))
            estimated_scale = scale
            estimated_location = real_camera_t + directional_vector * estimated_scale

            object.rotation_euler = Matrix(tranform_R @ transformed_rotation_euler.to_matrix()).to_euler("XYZ")
            object.location = estimated_location
            object.scale = (initial_scale * estimated_scale, initial_scale * estimated_scale, initial_scale * estimated_scale)

            object.keyframe_insert(data_path="rotation_euler", frame=idx + 1)
            object.keyframe_insert(data_path="location", frame=idx + 1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # visualize configuration
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--hmr_dir", type=str, default="results/generation/hmr")
    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--camera_motion_dir", type=str, default="results/generation/camera_motions")
    parser.add_argument("--scale_dir", type=str, default="results/generation/scale")
    parser.add_argument("--pnp_dir", type=str, default="results/generation/pnp")
    parser.add_argument("--idx", type=int, default=0)

    args = parser.parse_args()

    visualize_4d_hoi_sample(
        dataset=args.dataset,
        category=args.category,
        hmr_dir=args.hmr_dir,
        camera_dir=args.camera_dir,
        camera_motion_dir=args.camera_motion_dir,
        scale_dir=args.scale_dir,
        pnp_dir=args.pnp_dir,
        idx=args.idx,
    )