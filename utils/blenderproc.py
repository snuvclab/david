import blenderproc as bproc
import bpy
from bpy.app.handlers import persistent

import json
import numpy as np
from PIL import Image

# from constants.generation.assets import DATASET_PTHS, CATEGORY2DATASET_TYPE
from constants.visualizers import COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER


def initialize_scene(reset=False):
    if reset:
        bpy.ops.wm.read_factory_settings()

    bproc.utility.reset_keyframes()
    deleteListObjects = [
        "MESH",
        "CURVE",
        "SURFACE",
        "META",
        "FONT",
        "HAIR",
        "POINTCLOUD",
        "VOLUME",
        "GPENCIL",
        "ARMATURE",
        "LATTICE",
        "EMPTY",
        "LIGHT",
        "LIGHT_PROBE",
        "CAMERA",
        "SPEAKER",
    ]

    for o in bpy.context.scene.objects:
        for i in deleteListObjects:
            if o.type == i:
                o.select_set(False)
            else:
                o.select_set(True)

    bpy.ops.object.delete()

    for block in bpy.data.cameras:
        if block.users == 0:
            bpy.data.cameras.remove(block)


def set_render_config(resolution):
    scene = bpy.context.scene

    resolution_x_in_px, resolution_y_in_px = resolution
    scene.render.resolution_x = resolution_x_in_px
    scene.render.resolution_y = resolution_y_in_px
    scene.render.resolution_percentage = 100


def set_camera_config(scale=None, location=None, rotation=None):
    scene = bpy.context.scene
    camera = scene.camera
    cam_data = camera.data

    # default camera configurations
    # cam_data.type = "ORTHO"
    cam_data.type = "PERSP"
    cam_data.clip_start = 1
    cam_data.clip_end = 6000
    cam_data.sensor_width = 10
    cam_data.show_name = True
    camera.rotation_mode = "XYZ"

    # setting camera configurations
    if scale is not None:
        cam_data.ortho_scale = scale
    if location is not None:
        camera.location = location
    if rotation is not None:
        camera.rotation_euler = rotation

    bpy.context.view_layer.update()


def add_camera(resolution, f, name):
    scene = bpy.context.scene

    # setting render configuration
    set_render_config(resolution)

    # add camera
    bpy.ops.object.add(type="CAMERA")
    camera = bpy.context.object
    camera.name = f"{name}"
    cam_data = camera.data

    cam_data.name = name
    cam_data.lens = f
    scene.camera = camera

    return camera


def add_plane(scale=(10, 10, 10)):
    bpy.ops.mesh.primitive_plane_add(size=1000, location=(0, 0, 0), scale=scale)
    plane = bpy.context.selected_objects[0]

    return plane


def add_light(energy=50000, size=100, location=(0, 0, 10), rotaton=(0, 0, 0)):
    bpy.ops.object.light_add(type="AREA", location=location, rotation=rotaton)
    light = bpy.data.objects["Area"]
    light.data.energy = energy
    light.data.size = size

    return light


def add_hoi_samples(sample_path):
    bpy.ops.import_scene.obj(filepath=sample_path)
    asset_object = bpy.context.selected_objects[0]

    return asset_object


def add_assets(obj_path):
    # dataset_type = CATEGORY2DATASET_TYPE[(supercategory, category)]
    # dataset_path = DATASET_PTHS[dataset_type]

    # # different obj_path for dataset type
    # if dataset_type == "3D-FUTURE":
    #     obj_path = f"{dataset_path}/{asset_id}/raw_model.obj"

    # elif dataset_type == "SHAPENET":
    #     with open(f"{dataset_path}/taxonomy.json", "r") as file:
    #         categories = json.load(file)

    #     selected_category_info, *_ = [category_info for category_info in categories if category_info["name"] == category]
    #     obj_path = f"{dataset_path}/{selected_category_info['synsetId']}/{asset_id}/models/model_normalized.obj"

    # elif dataset_type == "SKETCHFAB" or dataset_type == "SAPIEN":
    #     obj_path = f"{dataset_path}/{supercategory}/{asset_id}/model.obj"

    # elif dataset_type == "BEHAVE":
    #     if disable_lowres_switch_for_behave:
    #         obj_path = f"{dataset_path}/objects/{category}/{category}.obj"
    #     else:
    #         obj_path = f"{dataset_path}/objects/{category}/{category}_canon_lowres_in_gen_coord.obj"

    # elif dataset_type == "INTERCAP":
    #     obj_path = f"{dataset_path}/objects/{category}/mesh.obj"

    # add asset
    bpy.ops.import_scene.obj(filepath=obj_path)
    asset_object = bpy.context.selected_objects[0]

    vertices = np.array([vertex.co for vertex in asset_object.data.vertices])  # N x 3
    faces = np.array([[vertex for vertex in polygon.vertices] for polygon in asset_object.data.polygons])  # M x 3

    # pre-rocseeing for "SHAPENET" dataset type
    # if place_on_floor and dataset_type in ["SHAPENET", "SKETCHFAB", "SAPIEN", "INTERCAP", "BEHAVE"]:
    #     z_min = vertices[:, 1].min()
    #     vertices -= [0.0, z_min, 0.0]
    #     asset_object.location.z -= z_min

    vertices = vertices @ COMPATIBILITY_MATRIX_TRIMESH_P3D_TO_BLENDER

    return asset_object, vertices, faces


def render(output_path, hide_objects, use_gpu=True, desc="", handler=None):
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.engine = "CYCLES" if use_gpu else "BLENDER_EEVEE"
    bpy.context.scene.cycles.device = "GPU" if use_gpu else "CPU"
    bpy.context.scene.render.filepath = output_path

    if handler is not None:
        bpy.app.handlers.render_complete.append(handler)

    for object in hide_objects:
        object.hide_render = True

    print(f"start rendering ... ({desc})")
    bpy.ops.render.render(write_still=True)

    for object in hide_objects:
        object.hide_render = False

    if handler is not None:
        bpy.app.handlers.render_complete.remove(handler)


def render_points(coordinates):
    compatibility_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    scene = bpy.context.scene
    camera = scene.camera
    camera_matrix_world = camera.matrix_world

    image_width = bpy.context.scene.render.resolution_x
    image_height = bpy.context.scene.render.resolution_y

    inv_camera_matrix_world = compatibility_matrix @ np.array(camera_matrix_world.inverted())[:3, :]  # 3 x 4
    point_in_camera_space = (inv_camera_matrix_world @ coordinates.transpose((0, 2, 1))).transpose((0, 2, 1))  # 3 x 4 @ N x 4 x 4 -> N x 3 x 4 -> N x 4 x 3

    proj_points = (point_in_camera_space[:, :, :2] * np.array([[image_width, image_height]]) / camera.data.ortho_scale) + np.array([[image_width / 2, image_height / 2]])  # N x 4 x 2

    return proj_points


def segmentation_handler(image_path):
    @persistent
    def extract_segmentation(dummy):
        Image.fromarray(np.array(Image.open(image_path).convert("RGBA"))[:, :, 3]).save(image_path)

    return extract_segmentation