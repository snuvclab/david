import numpy as np
import smplx
import torch
import pickle
import random
import argparse
from glob import glob
import open3d as o3d
import os
import cv2
from constants.david import SELECTED_INDICES

COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

HAND_INFO = dict(
    LEFT=1,
    RIGHT=2,
    NO=0
)

def prepare_dataset(
    dataset,
    category,
    hmr_dir,
    hoi_dir,
    hoi_data_dir,
    hand_info,
    skip_done
):
    COMPATIBILITY_MATRIX = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

    
    if hand_info == 1:
        hand_category = f"left_{category}"
    elif hand_info == 2:
        hand_category = f"right_{category}"
    elif hand_info == 0:
        hand_category = f"{category}"

    if SELECTED_INDICES.get(dataset, None) is not None and SELECTED_INDICES[dataset].get(category, None) is not None:
        selected_indices = SELECTED_INDICES[dataset][category]
    else: selected_indices = None

    hmr_pths_ = sorted(glob(f"{hmr_dir}/{dataset}/{category}/*/*/*.npz"))
    if selected_indices is not None:
        hmr_pths = [hmr_pths_[index] for index in selected_indices]
    else:
        hmr_pths = hmr_pths_

    with open("constants/sampled_human_indices.pkl", "rb") as handle:
        human_sampled_indices = pickle.load(handle)
    
    pts = []
    RT = []
    body_poses = []
    for hmr_pth in hmr_pths:

        print(hmr_pth)
        obj_data_pth = hmr_pth.replace(hmr_dir, hoi_dir).replace(".npz", ".pkl")

        with open(obj_data_pth, "rb") as handle:
            obj_data = pickle.load(handle)

        scale = obj_data["obj_scale"]
        obj_vertices = obj_data["obj_vertices"] * scale
        # obj_faces = obj_data["obj_faces"]
        obj_R = obj_data["obj_R"]
        obj_t = obj_data["obj_t"]

        motion_global = np.load(hmr_pth)
        
        frame_num = motion_global["poses"].shape[0]
        smplxmodel = smplx.create(model_path="imports/hmr4d/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).cuda()
        global_smplxmodel_output = smplxmodel(
            betas=torch.from_numpy(motion_global["betas"].reshape((1, 10))).repeat((frame_num, 1)).to('cuda').float(),
            global_orient=torch.from_numpy(motion_global["poses"][:, :3]).to('cuda').float(),
            body_pose=torch.from_numpy(motion_global["poses"][:, 3 : 3 + 21*3]).to('cuda').float(),
            left_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            right_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            transl=torch.from_numpy(motion_global["trans"][:, :3]).to('cuda').float(),
            expression=torch.zeros((frame_num, 10), device="cuda").float(),
            jaw_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            leye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            reye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_vertices = global_smplxmodel_output.vertices.to(torch.float64).cpu().numpy()

        for frame_vertices, human_euler, frame_obj_R, frame_obj_t, frame_pose in zip(global_vertices, motion_global["poses"][:, :3], obj_R, obj_t, motion_global["poses"][:, 3 : 3 + 21*3]):
            sampled_vertices = frame_vertices[human_sampled_indices]

            human_R, _ = cv2.Rodrigues(human_euler)
            mean = np.mean(sampled_vertices, axis=0)
            normalized_vertices = sampled_vertices - mean.reshape((1, 3)) # 1024 x 3
            fully_normalized_vertices = normalized_vertices @ (human_R.T).T

            compatible_frame_obj_R = human_R.T @ COMPATIBILITY_MATRIX @ frame_obj_R # 3 x 3
            compatible_frame_obj_t = human_R.T @ (COMPATIBILITY_MATRIX @ frame_obj_t - mean.reshape((3, 1))) # 3 x 1

            compatible_frame_obj_Rt = np.concatenate((compatible_frame_obj_R, compatible_frame_obj_t), axis=1) # 3 x 4

            pts.append(fully_normalized_vertices)
            RT.append(compatible_frame_obj_Rt)
            body_poses.append(frame_pose.reshape((21, 3)))

    pts = np.array(pts) # N x 1024 x 3
    RT = np.array(RT) # N x 3 x 4
    body_poses = np.array(body_poses) # N x 63 x 3

    to_save = dict(
        points=pts,
        transform=RT,
        poses=body_poses,
    )
    
    save_pth = f"{hoi_data_dir}/{dataset}/{hand_category}/RT.pkl"
    os.makedirs("/".join(save_pth.split("/")[:-1]), exist_ok=True)

    with open(save_pth, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="frypan")
    parser.add_argument("--hmr_dir", type=str, default="results/generation/hmr")
    parser.add_argument("--hoi_dir", type=str, default="results/generation/hoi")
    parser.add_argument("--hoi_data_dir", type=str, default="results/david/hoi_data")
    parser.add_argument("--hand_info", type=int, default=0)

    parser.add_argument("--skip_done", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    prepare_dataset(
        dataset=args.dataset,
        category=args.category,
        hmr_dir=args.hmr_dir,
        hoi_dir=args.hoi_dir,
        hoi_data_dir=args.hoi_data_dir,
        hand_info=args.hand_info,
        skip_done=args.skip_done
    )
