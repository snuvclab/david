import os
import pickle
import numpy as np
import torch
import sys
import trimesh

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from imports.genpose.configs.config import get_config
from imports.genpose.networks.posenet_agent_RT import PoseNet
from imports.genpose.utils.metrics import get_rot_matrix
import smplx
from tqdm import tqdm
import cv2
from glob import glob
import os
import argparse
from utils.dataset import category2object
import open3d as o3d
from constants.david import SELECTED_INDICES

def inference_omdm():
    cfg = get_config()
    cfg.batch_size = 1

    category = cfg.category
    dataset = cfg.dataset
    cfg.sampler_mode = ['ode_inference'] # override

    object_path = category2object(dataset, category)

    obj_mesh = trimesh.load(object_path, force='mesh')
    obj_vertices = np.array(obj_mesh.vertices).reshape((-1, 3))
    
    with open(sorted(glob(f"rebuttal/fullbodymanip/train_data/{category}/*/object/*.pkl"))[0], "rb") as handle:
        object_data = pickle.load(handle)
    average_scale = float(object_data["obj_scale"][0])

    print(f"average scale: {average_scale}")
    with open("constants/sampled_human_indices.pkl", "rb") as handle:
        human_sampled_indices = pickle.load(handle)

    motion_pths = sorted(glob(f"{cfg.inference_human_motion_dir}/{dataset}/{category}/*/*/*.npz"))

    for ii, motion_pth in enumerate(motion_pths[:]):
        save_pth = motion_pth.replace(cfg.inference_human_motion_dir, cfg.inference_object_motion_dir).replace(".npz", ".pkl")

        if cfg.skip_done and os.path.exists(save_pth): continue
        save_dir = "/".join(save_pth.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)

        generated_motion = np.load(motion_pth)
        frame_num = min(generated_motion["poses"].shape[0], generated_motion["trans"].shape[0])


        smplxmodel = smplx.create(model_path="imports/hmr4d/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).cuda()
        global_smplxmodel_output = smplxmodel(
            betas=torch.from_numpy(generated_motion["betas"].reshape((1, 10))).repeat((frame_num, 1)).to('cuda').float(),
            global_orient=torch.from_numpy(generated_motion["poses"][:frame_num, :3]).to('cuda').float(),
            body_pose=torch.from_numpy(generated_motion["poses"][:frame_num, 3 : 3 + 21*3]).to('cuda').float(),
            left_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            right_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            transl=torch.from_numpy(generated_motion["trans"][:frame_num, :3]).to('cuda').float(),
            expression=torch.zeros((frame_num, 10), device="cuda").float(),
            jaw_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            leye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            reye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_vertices = global_smplxmodel_output.vertices.to(torch.float64).cpu().numpy() # frame_num x 10475 x 3
        sampled_global_vertices = global_vertices[:, human_sampled_indices, :3] # frame_num x 1024 x 3
        global_joints = global_smplxmodel_output.joints.to(torch.float64).cpu()
        full_global_vertices = global_vertices
        

        t_pose_human = smplxmodel(
            betas=torch.from_numpy(generated_motion["betas"].reshape((1, 10))).repeat((frame_num, 1)).to('cuda').float(),
            global_orient=torch.from_numpy(generated_motion["poses"][:frame_num, :3]).to('cuda').float(),
            body_pose=torch.zeros_like(torch.from_numpy(generated_motion["poses"][:frame_num, 3 : 3 + 21*3])).to('cuda').float(),
            left_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            right_hand_pose=torch.zeros((frame_num, 45), device="cuda").float(),
            transl=torch.from_numpy(generated_motion["trans"][:frame_num, :3]).to('cuda').float(),
            expression=torch.zeros((frame_num, 10), device="cuda").float(),
            jaw_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            leye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            reye_pose=torch.zeros((frame_num, 3), device="cuda").float(),
            return_verts=True,
            return_full_pose=True,
        )
        t_pose_vertices = t_pose_human.vertices.to(torch.float64).cpu().numpy()
        height = t_pose_vertices[0, :, 1].max() - t_pose_vertices[0, :, 1].min() # N x 3
        ratio = height

        print(ratio)

        R = []
        t = []
        shifted_t = []

        score_agent = PoseNet(cfg)
        score_agent.load_ckpt(model_dir=cfg.score_model_path, model_path=True, load_model_only=True) 

        fully_normalized_sampled_global_vertices = []
        fully_normalized_full_global_vertices = []
        human_poses = []
        for ii, (full_global_vertex, frame_joints, sampled_global_vertex, human_euler, human_pose) in enumerate(tqdm(zip(full_global_vertices, global_joints, sampled_global_vertices, generated_motion["poses"][:frame_num, :3], generated_motion["poses"][:frame_num, 3 : 3 + 21*3]))):
            pelvis_joints = frame_joints[0]

            human_R, _ = cv2.Rodrigues(human_euler)
            mean = np.array(pelvis_joints) # 1 x 3
            normalized_sampled_global_vertex = sampled_global_vertex - mean.reshape((1, 3)) # 1024 x 3
            fully_normalized_sampled_global_vertex = normalized_sampled_global_vertex @ (human_R.T).T

            normalized_full_global_vertex = full_global_vertex - mean.reshape((1, 3)) # 1024 x 3
            fully_normalized_full_global_vertex = normalized_full_global_vertex @ (human_R.T).T

            fully_normalized_sampled_global_vertices.append(fully_normalized_sampled_global_vertex)
            fully_normalized_full_global_vertices.append(fully_normalized_full_global_vertex)

            human_poses.append(human_pose.reshape((21, 3)))

        frame_num = len(fully_normalized_sampled_global_vertices)
        fully_normalized_sampled_global_vertices = np.array(fully_normalized_sampled_global_vertices)
        fully_normalized_full_global_vertices = np.array(fully_normalized_full_global_vertices)
        human_poses = np.array(human_poses)

        pts = torch.from_numpy(fully_normalized_sampled_global_vertices).float().to("cuda")
        thetas = torch.from_numpy(human_poses).float().to("cuda")
        res_list, sampler_mode_list = score_agent.inference_score_func(
            batch_size=frame_num,
            thetas=thetas,
            human_vertices=fully_normalized_full_global_vertices,
            object_vertices=obj_vertices,
            ratio=ratio,
            object_scale=average_scale,
            contact_threshold=cfg.inference_contact_threshold, 
        )

        for i, sampler_mode in enumerate(sampler_mode_list):

            base_rot_mat = get_rot_matrix(res_list[i][:, :6], cfg.pose_mode).cpu().numpy() # frame_num x 3 x 3
            base_trans = res_list[i][:, 6:9].cpu().numpy() * ratio # frame_num x 3 x 1

        for frame_joints, rot_mat, trans, human_euler, sampled_global_vertex in zip(global_joints, base_rot_mat, base_trans, generated_motion["poses"][:frame_num, :3], sampled_global_vertices):
            pelvis_joints = frame_joints[0]
            
            human_R, _ = cv2.Rodrigues(human_euler)
            mean = np.array(pelvis_joints) # 1 x 3

            R.append(human_R @ rot_mat)
            t.append((human_R @ trans.reshape((3, 1)) + mean.reshape((3, 1))))

        with open(save_pth, 'wb') as f:
            pickle.dump(dict(
                R=np.array(R),
                t=np.array(t),
                shifted_t=np.array(shifted_t),
                average_scale=average_scale,
            ), f)


if __name__ == "__main__":
    inference_omdm()