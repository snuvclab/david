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
from constants.visualizers import COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER
from utils.dataset import category2object
from utils.blenderproc import initialize_scene, add_plane, add_light, add_camera, set_camera_config, render_points, render, segmentation_handler

def apply_shape_keys(ob):
    if not hasattr(ob.data, "shape_keys"):
        return
    ob.shape_key_add(name='CombinedKeys', from_mix=True)


def raycast(
    dataset,
    category,
    hmr_dir,
    object_vertices_dir,
    human_vertices_dir,
    camera_motion_dir,
    camera_motion_blender_dir,
    camera_dir,
    skip_done
):
    hmr_pths = glob(f"{hmr_dir}/{dataset}/{category}/*/*/*.npz")
    for hmr_pth in hmr_pths:
        # if args.skip_done and args.category is not None and category != args.category: continue

        object_path = category2object(dataset, category)
        camera_motion_path = hmr_pth.replace(hmr_dir, camera_motion_dir).replace(".npz", ".pkl")
        camera_motion_blender_save_path = hmr_pth.replace(hmr_dir, camera_motion_blender_dir).replace(".npz", ".pkl")
        object_vertices_save_path = hmr_pth.replace(hmr_dir, object_vertices_dir).replace(".npz", ".pkl")
        human_vertices_save_path = hmr_pth.replace(hmr_dir, human_vertices_dir).replace(".npz", ".pkl")

        object_vertices_save_dir = "/".join(object_vertices_save_path.split("/")[:-1])
        human_vertices_save_dir = "/".join(human_vertices_save_path.split("/")[:-1])
        camera_motion_blender_save_dir = "/".join(camera_motion_blender_save_path.split("/")[:-1])

        os.makedirs(object_vertices_save_dir, exist_ok=True)
        os.makedirs(human_vertices_save_dir, exist_ok=True)
        os.makedirs(camera_motion_blender_save_dir, exist_ok=True)

        if skip_done and os.path.exists(object_vertices_save_path) and os.path.exists(human_vertices_save_path): continue
            
        initialize_scene()
        
        # add smplx motion
        bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
        bpy.ops.object.smplx_add_animation(filepath=hmr_pth)
        smplx = bpy.data.objects['SMPLX-mesh-neutral']
        bpy.data.objects['SMPLX-mesh-neutral'].modifiers["Armature"].show_in_editmode = True
        bpy.data.objects['SMPLX-mesh-neutral'].modifiers["Armature"].show_on_cage = True
        bpy.data.objects['SMPLX-mesh-neutral'].modifiers["Armature"].use_deform_preserve_volume = True

        for shapeKey in smplx.data.shape_keys.key_blocks:
            smplx.shape_key_remove(shapeKey)

        bpy.ops.object.modifier_apply(modifier='Armature')

        bpy.context.view_layer.update()

        human_smplx_vertices = np.array([v_.co for v_ in smplx.data.vertices])

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

        with open(camera_motion_path, 'rb') as handle:
            camera_data = pickle.load(handle)

        R = camera_data['R']
        t = camera_data['t']
        
        camera_motion_blender_R = []
        camera_motion_blender_t = []
        for frame_idx in range(1, frame_num + 1):
            rotation_euler = Matrix(COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER.T @ R[frame_idx - 1]).to_euler("XYZ")
            location = t[frame_idx - 1].reshape((3,)) @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER

            rotation_euler[0] = np.pi + rotation_euler[0]
            camera.rotation_euler = rotation_euler
            camera.location = location.reshape((-1,))

            camera_motion_blender_R_frame = np.array(rotation_euler.to_matrix())
            camera_motion_blender_t_frame = np.array(location.reshape((-1,)))

            camera_motion_blender_R.append(camera_motion_blender_R_frame)
            camera_motion_blender_t.append(camera_motion_blender_t_frame)

            camera.keyframe_insert(data_path="rotation_euler", frame=frame_idx)
            camera.keyframe_insert(data_path="location", frame=frame_idx)

            if frame_idx == 1:
                initial_camera_rotation_euler_XYZ = rotation_euler
                initial_camera_location = location.reshape((-1,))

                initial_camera_rotation_euler = np.array(rotation_euler.to_matrix())
                initial_camera_translation = np.array(location).reshape((3, 1))

        camera_motion_blender = dict(
            R = np.array(camera_motion_blender_R),
            t = np.array(camera_motion_blender_t).reshape((-1, 3)),
        )

        with open(camera_motion_blender_save_path, "wb") as handle:
            pickle.dump(
                camera_motion_blender,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


        render_camera_pth = "/".join(hmr_pth.replace(hmr_dir, camera_dir).split("/")[:-2]) + ".pickle"
        with open(render_camera_pth, "rb") as handle:
            render_camera_data = pickle.load(handle)
        
        render_camera_R = np.array(render_camera_data["R"])
        render_camera_t = np.array(render_camera_data["t"]).reshape((3, 1))


        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]
        
        ## ADDED AFTER PERTURBATIOn SETTING
        if render_camera_data.get("obj_euler", None) is not None:
            object.rotation_euler = render_camera_data["obj_euler"].reshape((-1, ))
        if render_camera_data.get("obj_location", None) is not None:
            object.location = render_camera_data["obj_location"].reshape((-1, ))
        if render_camera_data.get("obj_scale", None) is not None:
            object.scale = render_camera_data["obj_scale"].reshape((-1, ))

        object_rotation = np.array(object.rotation_euler.to_matrix())
        object_translation = np.array(object.location).reshape((3, 1))

        tranform_R = initial_camera_rotation_euler @ np.linalg.inv(render_camera_R)
        transformed_t = tranform_R @ (object_translation - render_camera_t) + initial_camera_translation
        
        transformed_rotation_euler = Matrix(tranform_R @ object_rotation).to_euler("XYZ")
        transformed_location = transformed_t.reshape((-1, ))

        object.rotation_euler = transformed_rotation_euler
        object.location = transformed_location

        # Deselect mesh polygons and vertices
        def DeselectEdgesAndPolygons( obj ):
            for p in obj.data.polygons:
                p.select = False
            for e in obj.data.edges:
                e.select = False

        cam = camera
        cam.rotation_euler = initial_camera_rotation_euler_XYZ
        cam.location = initial_camera_location
        concerning_obj = object

        bpy.context.view_layer.update()
        limit = 0.005

        # Deselect mesh elements
        DeselectEdgesAndPolygons( concerning_obj )

        # In world coordinates, get a bvh tree and vertices
        vertices = [Matrix(tranform_R @ object_rotation) @ (v.co * object.scale[0]) + concerning_obj.location for v in concerning_obj.data.vertices]

        visible_vertices = []    
        for i, v in enumerate( tqdm (vertices)  ):
            # Get the 2D projection of the vertex
            co2D = world_to_camera_view( scene, cam, v )

            # If inside the camera view
            if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z >0: 
                # Try a ray cast, in order to test the vertex visibility from the camera
                location = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam.location, (v - cam.location).normalized() )
                # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
                if location[0] and (v - location[1]).length < limit:
                    visible_vertex = dict(vertices_2D=(WIDTH * co2D.x, HEIGHT * co2D.y), vertices_3D=(v[0], v[1], v[2]))
                    visible_vertices.append(visible_vertex)

        print(f"number of visible vertices: {len(visible_vertices)}")

        with open(object_vertices_save_path, "wb") as handle:
            pickle.dump(
                visible_vertices,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        
        human_visible_vertices = []    
        for i, v in enumerate( tqdm (human_smplx_vertices)  ):
            v = Vector(v)
            # Get the 2D projection of the vertex
            co2D = world_to_camera_view( scene, cam, v )

            # If inside the camera view
            if 0.0 <= co2D.x <= 1.0 and 0.0 <= co2D.y <= 1.0 and co2D.z >0: 
                # Try a ray cast, in order to test the vertex visibility from the camera
                location = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), cam.location, (v - cam.location).normalized() )
                # If the ray hits something and if this hit is close to the vertex, we assume this is the vertex
                if location[0] and (v - location[1]).length < limit:
                    hand_head_visible_vertex = dict(vertices_2D=(WIDTH * co2D.x, HEIGHT * co2D.y), vertices_3D=(v[0], v[1], v[2]))
                    human_visible_vertices.append(hand_head_visible_vertex)

        with open(human_vertices_save_path, "wb") as handle:
            pickle.dump(
                human_visible_vertices,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # visualize configuration
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--hmr_dir", type=str, default="results/generation/hmr")

    parser.add_argument("--object_vertices_dir", type=str, default="results/generation/object_vertices")
    parser.add_argument("--human_vertices_dir", type=str, default="results/generation/human_vertices")
    parser.add_argument("--camera_motion_dir", type=str, default="results/generation/camera_motions")
    parser.add_argument("--camera_motion_blender_dir", type=str, default="results/generation/camera_motions_blender")

    parser.add_argument("--camera_dir", type=str, default="results/generation/cameras")
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()

    raycast(
        dataset=args.dataset,
        category=args.category,
        hmr_dir=args.hmr_dir,
        object_vertices_dir=args.object_vertices_dir,
        human_vertices_dir=args.human_vertices_dir,
        camera_motion_dir=args.camera_motion_dir,
        camera_motion_blender_dir=args.camera_motion_blender_dir,
        camera_dir=args.camera_dir,
        skip_done=args.skip_done
    )