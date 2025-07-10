import blenderproc as bproc
import bpy
from mathutils import Matrix, Vector, Euler

import argparse
import numpy as np
from glob import glob
import pickle
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

import os
import sys
from tqdm import tqdm
import smplx
from smplx.utils import SMPLXOutput
sys.path.append(os.getcwd())
from utils.blenderproc import initialize_scene, add_plane, add_light
from utils.coap import attach_coap
from utils.dataset import category2object

INITIAL_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
width = 1200
height = 800
focal_length = np.sqrt(width**2 + height**2)

def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device=device).float()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device).float()
    return x

@torch.no_grad()
def sample_scene_points(model, smpl_output, scene_vertices, scene_normals=None, n_upsample=2, max_queries=10000):
    points = scene_vertices.clone()
    # remove points that are well outside the SMPL bounding box
    bb_min = smpl_output.vertices.min(1).values.reshape(1, 3)
    bb_max = smpl_output.vertices.max(1).values.reshape(1, 3)

    inds = (scene_vertices >= bb_min).all(-1) & (scene_vertices <= bb_max).all(-1)
    if not inds.any():
        return None
    points = scene_vertices[inds, :]
    model.coap.eval()
    colliding_inds = (model.coap.query(points.reshape((1, -1, 3)), smpl_output) > 0.01).reshape(-1)
    model.coap.detach_cache()  # detach variables to enable differentiable pass in the opt. loop
    if not colliding_inds.any():
        return None
    points = points[colliding_inds.reshape(-1)]
    
    if scene_normals is not None and points.size(0) > 0:  # sample extra points if normals are available
        norms = scene_normals[inds][colliding_inds]

        offsets = 0.5*torch.normal(0.05, 0.05, size=(points.shape[0]*n_upsample, 1), device=norms.device).abs()
        verts, norms = points.repeat(n_upsample, 1), norms.repeat(n_upsample, 1)
        points = torch.cat((points, (verts - offsets*norms).reshape(-1, 3)), dim=0)

    if points.shape[0] > max_queries:
        points = points[torch.randperm(points.size(0), device=points.device)[:max_queries]]

    print(points.shape)
    return points.float().reshape(1, -1, 3)  # add batch dimension


def minimum_distance(vertsA, vertsB, num_vertices=2000):
    # batch_size = 1 + vertsB.shape[1] // 10000
    # vertsB_batches = vertsB.split(batch_size)

    distances_A_to_B = []
    # for vertsB_batch in vertsB_batches:
    distances = torch.cdist(vertsA.float(), vertsB.float()) ** 2
    distances_A_to_B.append(distances)

    distances_A_to_B = torch.cat(distances_A_to_B, dim=1)
    min_distance_A_to_B = torch.min(distances_A_to_B, dim=1).values
    sorted_min_distance_A_to_B, _ = torch.sort(min_distance_A_to_B)
    clip_sorted_min_distance_A_to_B = torch.clip(sorted_min_distance_A_to_B[:num_vertices] - 0.0, min=0)
    distance = torch.mean(clip_sorted_min_distance_A_to_B)

    print(distance)
    return distance

def get_relaxed_hand_pose():
    data_path = "constants/smplx_handposes.npz"
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
        # hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(1, -1)

    return left_hand_pose, right_hand_pose


def in_mask(mask, x, y):
    h, w = mask.shape

    # Get integer pixel indices
    x0, x1 = int(np.floor(x)), int(np.ceil(x))
    y0, y1 = int(np.floor(y)), int(np.ceil(y))

    # Ensure indices are within bounds
    if x0 < 0 or x1 >= w or y0 < 0 or y1 >= h:
        return False

    # Get mask values at the 4 nearest integer pixels
    q11 = mask[y0, x0] if (0 <= x0 < w and 0 <= y0 < h) else 0
    q21 = mask[y0, x1] if (0 <= x1 < w and 0 <= y0 < h) else 0
    q12 = mask[y1, x0] if (0 <= x0 < w and 0 <= y1 < h) else 0
    q22 = mask[y1, x1] if (0 <= x1 < w and 0 <= y1 < h) else 0

    # Compute bilinear interpolation
    fx1 = (x1 - x) * q11 + (x - x0) * q21
    fx2 = (x1 - x) * q12 + (x - x0) * q22
    interpolated_value = (y1 - y) * fx1 + (y - y0) * fx2

    return interpolated_value > 0.5  # Threshold to determine inside/outside


def plot_points(coords, width=1200, height=800, dot_size=5, save_path=None):
    """
    Plots given (x, y) coordinates on a blank image.
    
    Parameters:
        coords (numpy.ndarray): Nx2 array of (x, y) coordinates.
        width (int): Width of the image.
        height (int): Height of the image.
        dot_size (int): Size of the dots.
        save_path (str): If provided, saves the image instead of displaying it.
    """
    coords = np.array(coords)

    # Create a blank white background
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Plot the points
    x, y = coords[:, 0], coords[:, 1]
    ax.scatter(x, y, color='red', s=dot_size)  # s = size of dots

    # Save or display
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def min_n_indices(tensor, N):
    """
    Returns the indices of the minimum N elements in a tensor.
    """
    return torch.topk(tensor, k=N, largest=True).indices


def MSE_loss(A, B):
    # loss = torch.nn.functional.mse_loss(A, B)
    loss = torch.nn.functional.l1_loss(A, B)
    return loss

def huber_loss(A, B, delta=1e-2):
    """
    Computes the Huber loss between two point clouds A and B.
    """
    diff = A - B
    abs_diff = torch.norm(diff, dim=1, p=2)  # Euclidean distance per point
    quadratic = 0.5 * abs_diff ** 2
    linear = delta * (abs_diff - 0.5 * delta)
    loss = torch.where(abs_diff < delta, quadratic, linear)  # Huber Loss formula
    return loss.mean()


def get_object_path(category):
    if category == "cart":
        object_path = "data/HOIs/cart/objects/00003.obj"
    elif category == "guitar":
        object_path = "data/ShapeNetCore.v2/03467517/5a5469a9912a03a0da5505f71d8f8d5b/models/model_normalized.obj"
    elif category == "motorcycle":
        object_path = "data/ShapeNetCore.v2/03790512/9b9794dda0a6532215a11c390f7ca182/models/model_normalized.obj"
    elif category in ["mop_", "vacuum_"]:
        object_path = glob(f"data/FullBodyManip/{category}/model.obj")[0]
    elif len(glob(f"data/*/{category}/*/model.obj")) > 0:
        object_path = glob(f"data/*/{category}/*/model.obj")[0]
    elif len(glob(f"data/*/*/{category}/*/model.obj")) > 0:
        object_path = glob(f"data/*/*/{category}/*/model.obj")[0]
    elif len(glob(f"data/INTERCAP/objects/{category}/mesh.obj")) > 0:
        object_path = glob(f"data/INTERCAP/objects/{category}/mesh.obj")[0]
    elif len(glob(f"data/BEHAVE/objects/{category}/{category}.obj")) > 0:
        object_path = glob(f"data/BEHAVE/objects/{category}/{category}.obj")[0]
    
    return object_path

def optimize_RTS(
    dataset,
    category,
    vertices_dir,
    human_vertices_dir,
    rgb_dir,
    depth_dir,
    seg_dir,
    depthmap_dir,
    pnp_dir,
    final_scale_dir,
    camera_motion_blender_dir,
    render_camera_dir,
    iteration,
    skip_done
): 

    vertices_pths = sorted(glob(f"{vertices_dir}/{dataset}/{category}/*/*/*.pkl"))

    for vertex_pth in vertices_pths[:]:
        initialize_scene()
        
        bpy.ops.object.light_add(type="AREA", location=(0.0, 0.0, 5.0), rotation=(0.0, 0.0, 0.0))
        light = bpy.data.objects["Area"]
        light.data.energy = 3000
        light.data.size = 2

        add_plane()

        scene = bpy.context.scene
        scene.render.resolution_x = width
        scene.render.resolution_y = height
        scene.render.resolution_percentage = 100

        bpy.ops.object.add(type="CAMERA")
        camera = bpy.context.object
        camera.name = "camera"
        cam_data = camera.data
        cam_data.name = "camera"
        cam_data.sensor_width = 10
        cam_data.lens = np.sqrt(width**2 + height**2) * cam_data.sensor_width / max(width, height)
        scene.camera = camera

        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.engine = "CYCLES"
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = 4.0
        camera.location = (-0.131889, 1.85043, 1.45107)
        camera.rotation_euler = (73.6 / 180 * np.pi, 0.0, np.pi)


        smplx_pose_pth = vertex_pth.replace("object_vertices", "hmr").replace(".pkl", ".npz")
        bpy.ops.preferences.addon_enable(module="smplx_blender_addon")
        bpy.ops.object.smplx_add_animation(filepath=smplx_pose_pth)

        category = vertex_pth.split("/")[-4]
        view_id = vertex_pth.split("/")[-3]

        if args.category is not None and category != args.category: continue

        rgb_pth = "/".join(vertex_pth.replace(vertices_dir, rgb_dir).split("/")[:-1]) + ".png"
        human_vertex_pth = vertex_pth.replace(vertices_dir, human_vertices_dir)
        depth_pth = vertex_pth.replace(vertices_dir, depth_dir)
        # seg_pth = vertex_pth.replace(vertices_dir, seg_dir).replace(".pkl", ".png")
        camera_motion_blender_pth = vertex_pth.replace(vertices_dir, camera_motion_blender_dir)
        render_camera_pth = f"results/generation/cameras/{dataset}/{category}/{view_id}.pickle"
        pnp_pth = vertex_pth.replace(vertices_dir, pnp_dir)
        save_pth = vertex_pth.replace(vertices_dir, final_scale_dir)
        save_dir = "/".join(save_pth.split("/")[:-1])

        if skip_done and os.path.exists(save_pth): continue
        os.makedirs(save_dir, exist_ok=True)


        try:
            with open(vertex_pth, "rb") as handle: object_vertex_data = pickle.load(handle)
            with open(human_vertex_pth, "rb") as handle: human_vertex_data = pickle.load(handle)
            with open(depth_pth, "rb") as handle: depth_data = pickle.load(handle)
            with open(camera_motion_blender_pth, "rb") as handle: camera_motion_data = pickle.load(handle)
            with open(pnp_pth, "rb") as handle: pnp_data = pickle.load(handle)
            with open(render_camera_pth, "rb") as handle: render_camera_data = pickle.load(handle)
        except:
            continue
        smplx_pose_data = np.load(smplx_pose_pth)

        object_vertex_2D = [vertex_data["vertices_2D"] for vertex_data in object_vertex_data]
        object_vertex_3D = [vertex_data["vertices_3D"] for vertex_data in object_vertex_data]
        human_vertex_2D = [vertex_data["vertices_2D"] for vertex_data in human_vertex_data]
        human_vertex_3D = [vertex_data["vertices_3D"] for vertex_data in human_vertex_data]
        camera_translation = camera_motion_data["t"][0]
        camera_rotation = camera_motion_data["R"][0]
        render_camera_R = np.array(render_camera_data["R"])
        render_camera_t = np.array(render_camera_data["t"]).reshape((3, 1))
        initial_object_scale = render_camera_data["obj_scale"].reshape((-1, ))

        print(f"object vertex 2D shape: {np.array(object_vertex_2D).shape}")
        print(f"object vertex 3D shape: {np.array(object_vertex_3D).shape}")
        print(f"human vertex 2D shape: {np.array(human_vertex_2D).shape}")
        print(f"human vertex 3D shape: {np.array(human_vertex_3D).shape}")
        # Depth Guidance        
        depth = torch.from_numpy(depth_data["depth"]).unsqueeze(0).unsqueeze(0).to("cuda")

        object_depths = []
        for idx, vertex_2D in enumerate(object_vertex_2D):
            vertex_2D = torch.tensor([vertex_2D[0] / (width - 1) * 2 - 1, (height - vertex_2D[1]) / (height - 1) * 2 - 1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float().to("cuda")
            z = torch.nn.functional.grid_sample(depth, vertex_2D, align_corners=True, mode="bilinear") # 1 x 1 x 1 x 1
            object_depths.append(float(z.cpu().numpy()))
        object_depths = np.array(object_depths).reshape((-1, len(object_depths)))

        x, y = np.array(object_vertex_2D)[:, 0], np.array(object_vertex_2D)[:, 1]
        x, y = x.reshape((-1, len(x))), y.reshape((-1, len(y)))
        x, y = -(x - width / 2) / focal_length, (y - height / 2) / focal_length
        object_points = np.stack((np.multiply(x, object_depths), np.multiply(y, object_depths), object_depths), axis=-1).reshape(-1, 3)

        human_depths = []
        for idx, vertex_2D in enumerate(human_vertex_2D):
            vertex_2D = torch.tensor([vertex_2D[0] / (width - 1) * 2 - 1, (height - vertex_2D[1]) / (height - 1) * 2 - 1]).unsqueeze(0).unsqueeze(0).unsqueeze(0).float().to("cuda")
            z = torch.nn.functional.grid_sample(depth, vertex_2D, align_corners=True, mode="bilinear") # 1 x 1 x 1 x 1
            human_depths.append(float(z.cpu().numpy()))
        human_depths = np.array(human_depths).reshape((-1, len(human_depths)))

        x, y = np.array(human_vertex_2D)[:, 0], np.array(human_vertex_2D)[:, 1]
        x, y = x.reshape((-1, len(x))), y.reshape((-1, len(y)))
        x, y = -(x - width / 2) / focal_length, (y - height / 2) / focal_length
        human_points = np.stack((np.multiply(x, human_depths), np.multiply(y, human_depths), human_depths), axis=-1).reshape(-1, 3)

        print(f"object depth shape: {object_depths.shape}")
        print(f"human depth shape: {human_depths.shape}")

        print(f"object points shape: {object_points.shape}")
        print(f"human points shape: {human_points.shape}")

        canonical_camera_R = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]).reshape((3, 3))
        canonical_camera_t = np.array([0.0, 0.0, 0.0])

        transformed_human_points = (camera_motion_data["R"][0] @ np.linalg.inv(canonical_camera_R) @ human_points.T + camera_motion_data["t"][0].reshape((3, 1))).T
        transformed_object_points = (camera_motion_data["R"][0] @ np.linalg.inv(canonical_camera_R) @ object_points.T + camera_motion_data["t"][0].reshape((3, 1))).T

        print(transformed_human_points[:, 1].min())
        print(transformed_human_points[:, 1].max())
        print(transformed_object_points[:, 1].min())
        print(transformed_object_points[:, 1].max())

        print(f"canonical camera R: {canonical_camera_R}")

        # PnP Guidance
        initial_camera_R = camera_motion_data["R"][0]
        initial_camera_t = camera_motion_data["t"][0].reshape((3, 1))

        object_path = category2object(dataset, category)
        bpy.ops.import_scene.obj(filepath=object_path)
        object = bpy.context.selected_objects[0]
        object_data = bpy.context.selected_objects[0].data
        object.scale = initial_object_scale

        if render_camera_data.get("obj_euler", None) is not None:
            initial_object_R = render_camera_data["obj_R"].reshape((3, 3))
        if render_camera_data.get("obj_location", None) is not None:
            initial_object_t = render_camera_data["obj_t"].reshape((3, 1))
            
        print(f"initial object R: {initial_object_R}")


        for idx, pnp_result in enumerate(pnp_data):
            HMR_camera_R = camera_motion_data["R"][idx]
            HMR_camera_t = camera_motion_data["t"][idx]

            pnp_cam_R = pnp_result["inferred_cam_R"] # 3 x 3
            pnp_cam_t = pnp_result["inferred_cam_t"] # 3 x 1
            
            real_camera_R = pnp_result["real_camera_R"] # 3 x 3
            real_camera_t = pnp_result["real_camera_t"] # 3 x 1

            initial_tranform_R = initial_camera_R @ np.linalg.inv(render_camera_R) # 3 x 3
            initial_transformed_t = initial_tranform_R @ (initial_object_t - render_camera_t) + initial_camera_t

            object_transform_R = (-real_camera_R @ np.linalg.inv(pnp_cam_R))
            object_transform_t = object_transform_R @ (initial_transformed_t.reshape((3, 1)) - pnp_cam_t) + real_camera_t
            
            if idx == 0:

                human = smplx.create(model_path="imports/hmr4d/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45)
                human = attach_coap(human, pretrained=True, device='cuda')

                left_hand_pose, right_hand_pose = get_relaxed_hand_pose()
                smplx_cam_output = human(
                    body_pose=to_tensor(smplx_pose_data["poses"][0, 1 * 3:1 * 3 + 21 * 3].reshape((-1, 63)), 'cuda'),
                    betas=to_tensor(smplx_pose_data["betas"].reshape((1, 10)), 'cuda'),
                    global_orient=to_tensor(smplx_pose_data["poses"][0, :1 * 3].reshape((-1, 3)), 'cuda'),
                    transl=to_tensor(smplx_pose_data["trans"][0, :1 * 3].reshape((-1, 3)), 'cuda'),
                    expression=to_tensor(np.zeros((1, 10)), 'cuda'),
                    jaw_pose=to_tensor(np.zeros((1, 3)), 'cuda'),
                    leye_pose=to_tensor(np.zeros((1, 3)), 'cuda'),
                    reye_pose=to_tensor(np.zeros((1, 3)), 'cuda'),
                    left_hand_pose=to_tensor(left_hand_pose.reshape((1, 45)), 'cuda'),
                    right_hand_pose=to_tensor(right_hand_pose.reshape((1, 45)), 'cuda'),
                    return_verts=True,
                    return_full_pose=True
                )
                full_human_vertices = (smplx_cam_output.vertices @ torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T.to('cuda'))
                full_human_joints = (smplx_cam_output.joints @ torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T.to('cuda'))
                full_object_vertices = torch.tensor([Matrix(object_transform_R @ initial_tranform_R @ initial_object_R) @ (v.co * object.scale[0]) + Vector(object_transform_t) for v in object_data.vertices]).to('cuda')

                human_scale = torch.nn.Parameter(torch.tensor([1.0]).to("cuda"), requires_grad=True).float()
                object_scale = torch.nn.Parameter(torch.tensor([1.0]).to("cuda"), requires_grad=True).float()
                optimizer = torch.optim.Adam([human_scale, object_scale], lr=3.5e-3)

                cam_center = torch.tensor(HMR_camera_t.reshape((3, 1))).to("cuda").float() # 3 x 1
                human_direction_vector = torch.tensor(np.array(human_vertex_3D).T - HMR_camera_t.reshape((3, 1))).to("cuda") # 3 x N
                object_direction_vector = torch.tensor(np.array(object_vertex_3D).T - HMR_camera_t.reshape((3, 1))).to("cuda") # 3 x M
                human_depthmap_GT = torch.tensor(transformed_human_points.T).to("cuda")
                object_depthmap_GT = torch.tensor(transformed_object_points.T).to("cuda")


                full_human_verts_direction_vector = (full_human_vertices.squeeze(0) - torch.tensor(HMR_camera_t.reshape((1, 3))).to("cuda")).float() # N x 3
                full_human_joints_direction_vector = (full_human_joints.squeeze(0) - torch.tensor(HMR_camera_t.reshape((1, 3))).to("cuda")).float() # N x 3
                full_object_direction_vector = (full_object_vertices - torch.tensor(HMR_camera_t.reshape((1, 3))).to("cuda")).float() # M x 3

                for i in tqdm(range(iteration)):
                    optimizer.zero_grad()

                    print(float(object_scale.cpu().float() / human_scale.cpu().float()))
                    if i >= 2000:
                        new_full_human_vertices = cam_center.reshape((1, 3)) + full_human_verts_direction_vector * human_scale 
                        new_full_human_joints = cam_center.reshape((1, 3)) + full_human_joints_direction_vector * human_scale 
                        new_full_object_vertices = cam_center.reshape((1, 3)) + full_object_direction_vector * object_scale

                        smplx_real_output = SMPLXOutput(
                            vertices=new_full_human_vertices.unsqueeze(0),
                            joints=new_full_human_joints.unsqueeze(0),
                            betas=smplx_cam_output.betas.clone().detach(),
                            expression=smplx_cam_output.expression.clone().detach(),
                            global_orient=smplx_cam_output.global_orient.clone().detach(),
                            body_pose=smplx_cam_output.body_pose.clone().detach(),
                            left_hand_pose=smplx_cam_output.left_hand_pose.clone().detach(),
                            right_hand_pose=smplx_cam_output.right_hand_pose.clone().detach(),
                            jaw_pose=smplx_cam_output.jaw_pose.clone().detach(),
                            v_shaped=smplx_cam_output.v_shaped.clone().detach(),
                            full_pose=smplx_cam_output.full_pose.clone().detach()
                        )

                        asset_points = sample_scene_points(human, smplx_real_output, new_full_object_vertices.unsqueeze(0))
                        collision_loss = 1e-1 * (human.coap.collision_loss(asset_points, smplx_real_output)[0] if asset_points is not None else 0.0)
                        contact_loss = 1e3 * minimum_distance(new_full_object_vertices, new_full_human_vertices, 5000)

                    new_human_vertex_3D = cam_center + human_direction_vector * human_scale 
                    new_object_vertex_3D = cam_center + object_direction_vector * object_scale

                    human_loss = huber_loss(new_human_vertex_3D, human_depthmap_GT)
                    object_loss = huber_loss(new_object_vertex_3D, object_depthmap_GT)
                    
                    if i >= 2000:
                        total_loss = collision_loss + contact_loss
                        total_loss.backward()
                    else:
                        total_loss = human_loss + object_loss
                        total_loss.backward()

                    optimizer.step()

                relative_scale = float(object_scale.cpu().float() / human_scale.cpu().float())

                print(f"optimized human scale: {human_scale.cpu().float()}")
                print(f"optimized object scale: {object_scale.cpu().float()}")
                print(f"relative_scale: {relative_scale}")

            object.rotation_euler = Matrix(object_transform_R @ initial_tranform_R @ initial_object_R).to_euler("XYZ")
            optimized_object_location = HMR_camera_t.reshape((3, 1)) + (object_transform_t.reshape((3, 1)) - HMR_camera_t.reshape((3, 1))) * relative_scale
            object.location = optimized_object_location.reshape((-1, ))
            object.scale = (relative_scale, relative_scale, relative_scale)
            object.keyframe_insert(data_path="rotation_euler", frame=idx + 1)
            object.keyframe_insert(data_path="location", frame=idx + 1)

            with open(save_pth, "wb") as handle:
                pickle.dump(
                    dict(
                        scale=float(relative_scale)
                    ),
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")

    parser.add_argument("--vertices_dir", type=str, default="results/generation/object_vertices")
    parser.add_argument("--human_vertices_dir", type=str, default="results/generation/human_vertices")

    parser.add_argument("--depth_dir", type=str, default="results/generation/depth")
    parser.add_argument("--seg_dir", type=str, default="results/generation/humanseg")
    parser.add_argument("--depthmap_dir", type=str, default="results/generation/depthmap")
    parser.add_argument("--pnp_dir", type=str, default="results/generation/pnp")
    parser.add_argument("--final_scale_dir", type=str, default="results/generation/scale")

    parser.add_argument("--rgb_dir", type=str, default="results/generation/images")
    parser.add_argument("--camera_motion_blender_dir", type=str, default="results/generation/camera_motions_blender")
    parser.add_argument("--render_camera_dir", type=str, default="results/generation/cameras")
    
    parser.add_argument("--iteration", default=3000)
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    optimize_RTS(
        dataset=args.dataset,
        category=args.category,
        vertices_dir=args.vertices_dir,
        human_vertices_dir=args.human_vertices_dir,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        seg_dir=args.seg_dir,
        depthmap_dir=args.depthmap_dir,
        pnp_dir=args.pnp_dir,
        final_scale_dir=args.final_scale_dir,
        camera_motion_blender_dir=args.camera_motion_blender_dir,
        render_camera_dir=args.render_camera_dir,
        iteration=args.iteration,
        skip_done=args.skip_done
    )