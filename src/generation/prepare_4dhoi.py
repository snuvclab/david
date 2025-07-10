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
from constants.config import WIDTH, HEIGHT
from constants.visualizers import COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER
from utils.dataset import category2object

def prepare_4dhoi(
    dataset,
    category,
    hmr_dir,
    camera_dir,
    camera_motion_dir,
    scale_dir,
    pnp_dir,
    hoi_dir,
    skip_done,
):
    hmr_pths = glob(f"{hmr_dir}/{dataset}/{category}/*/*/*.npz")

    for hmr_pth in hmr_pths:
        
        category = hmr_pth.split("/")[-4]
        if args.category is not None and category != args.category: continue

        camera_motion_pth = hmr_pth.replace(hmr_dir, camera_motion_dir).replace(".npz", ".pkl")
        scale_pth = hmr_pth.replace(hmr_dir, scale_dir).replace(".npz", ".pkl")
        pnp_pth = hmr_pth.replace(hmr_dir, pnp_dir).replace(".npz", ".pkl")
        hoi_pth = hmr_pth.replace(hmr_dir, hoi_dir).replace(".npz", ".pkl")
        hoi_save_dir = "/".join(hoi_pth.split("/")[:-1])

        os.makedirs(hoi_save_dir, exist_ok=True)
        if skip_done and os.path.exists(hoi_pth): continue

        initialize_scene()
        
        # add smplx motion
        bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
        bpy.ops.object.smplx_add_animation(filepath=hmr_pth)

        smplx_poses = np.load(hmr_pth)
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

        if not os.path.exists(pnp_pth) or not os.path.exists(camera_motion_pth) or not os.path.exists(scale_pth): continue

        with open(camera_motion_pth, 'rb') as handle:
            camera_data = pickle.load(handle)\

        with open(pnp_pth, 'rb') as handle:
            pnp_results = pickle.load(handle)

        R = camera_data['R']
        t = camera_data['t']

        for frame_idx in range(1, frame_num + 1):
            rotation_euler = Matrix(COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER.T @ R[frame_idx - 1]).to_euler("XYZ")
            location = t[frame_idx - 1].reshape((3,)) @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER

            rotation_euler[0] = np.pi + rotation_euler[0]
            camera.rotation_euler = rotation_euler
            camera.location = location.reshape((-1,))

            camera.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            camera.keyframe_insert(data_path="location", frame=frame_idx)

            if frame_idx == 1:
                initial_camera_rotation_euler = np.array(camera.rotation_euler.to_matrix())
                initial_camera_translation = np.array(camera.location).reshape((3, 1))

        render_camera_pth = "/".join(hmr_pth.replace(hmr_dir, camera_dir).split("/")[:-2]) + ".pickle"
        with open(render_camera_pth, "rb") as handle:
            render_camera_data = pickle.load(handle)
        
        render_camera_R = np.array(render_camera_data["R"])
        render_camera_t = np.array(render_camera_data["t"]).reshape((3, 1))

        object_path = category2object(dataset, category)
        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]

        initial_scale = 1.0
        if render_camera_data.get("obj_euler", None) is not None:
            object.rotation_euler = render_camera_data["obj_euler"].reshape((-1, ))
        if render_camera_data.get("obj_location", None) is not None:
            object.location = render_camera_data["obj_location"].reshape((-1, ))
        if render_camera_data.get("obj_scale", None) is not None:
            object.scale = render_camera_data["obj_scale"].reshape((-1, ))
            initial_scale = float(render_camera_data["obj_scale"][0])

        object_rotation = np.array(object.rotation_euler.to_matrix())
        object_translation = np.array(object.location).reshape((3, 1))

        tranform_R = initial_camera_rotation_euler @ np.linalg.inv(render_camera_R)
        transformed_t = tranform_R @ (object_translation - render_camera_t) + initial_camera_translation
        
        transformed_rotation_euler = Matrix(tranform_R @ object_rotation).to_euler("XYZ")
        transformed_location = transformed_t.reshape((-1, ))

        object.rotation_euler = transformed_rotation_euler
        object.location = transformed_location
        directional_vector = transformed_location.reshape((3, 1)) - initial_camera_translation.reshape((3, 1))

        bpy.context.view_layer.update()

        with open(scale_pth, "rb") as handle:
            scale_data = pickle.load(handle)

        scale = scale_data["scale"]

        obj_R = []
        obj_t = []
        for idx, pnp_result in enumerate(pnp_results):
            pnp_cam_R = pnp_result["inferred_cam_R"] # 3 x 3
            pnp_cam_t = pnp_result["inferred_cam_t"] # 3 x 1
            
            real_camera_R = pnp_result["real_camera_R"] # 3 x 3
            real_camera_t = pnp_result["real_camera_t"] # 3 x 1

            tranform_R = -real_camera_R @ np.linalg.inv(pnp_cam_R)
            transformed_t = tranform_R @ (transformed_location.reshape((3, 1)) - pnp_cam_t) + real_camera_t

            directional_vector = transformed_t.reshape((3, 1)) - real_camera_t.reshape((3, 1))
            estimated_scale = scale
            estimated_location = real_camera_t + directional_vector * estimated_scale

            object.rotation_euler = Matrix(tranform_R @ transformed_rotation_euler.to_matrix()).to_euler("XYZ")
            object.location = estimated_location
            object.scale = (initial_scale * estimated_scale, initial_scale * estimated_scale, initial_scale * estimated_scale)

            obj_R.append(np.array(tranform_R @ transformed_rotation_euler.to_matrix()).reshape((3, 3)))
            obj_t.append(np.array(estimated_location).reshape((3, 1)))

            object.keyframe_insert(data_path="rotation_euler", frame=idx + 1)
            object.keyframe_insert(data_path="location", frame=idx + 1)

        me = object.data
        bm = bmesh.new()
        bm.from_mesh(me)
        object_vertices = np.array([v_.co for v_ in bm.verts])
        object_faces = np.array([[v.index for v in f.verts] for f in bm.faces])

        with open(hoi_pth, "wb") as handle:
            pickle.dump(
                dict(
                    obj_R=obj_R,
                    obj_t=obj_t,
                    obj_vertices=object_vertices,
                    obj_faces=object_faces,
                    obj_scale=scale,
                ),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--hmr_dir", type=str, default="results/generation/hmr")
    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--camera_motion_dir", type=str, default="results/generation/camera_motions")
    parser.add_argument("--scale_dir", type=str, default="results/generation/scale")
    parser.add_argument("--pnp_dir", type=str, default="results/generation/pnp")
    parser.add_argument("--hoi_dir", type=str, default="results/generation/hoi")
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()

    prepare_4dhoi(
        dataset=args.dataset,
        category=args.category,
        hmr_dir=args.hmr_dir,
        camera_dir=args.camera_dir,
        camera_motion_dir=args.camera_motion_dir,
        scale_dir=args.scale_dir,
        pnp_dir=args.pnp_dir,
        hoi_dir=args.hoi_dir,
        skip_done=args.skip_done,
    )

    