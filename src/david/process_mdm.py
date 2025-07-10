import sys, os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
import smplx
import argparse

from imports.mdm.human_body_prior.tools.omni_tools import copy2cpu as c2c
from imports.mdm.human_body_prior.body_model.body_model import BodyModel
from os.path import join as pjoin

from imports.mdm.common.skeleton import Skeleton
import numpy as np
import os
from imports.mdm.common.quaternion import *
from imports.mdm.paramUtil import *
import spacy
import os
import random
from constants.david import SELECTED_INDICES
import pickle
import json

nlp = spacy.load("en_core_web_sm")

import torch
from tqdm import tqdm
import os
from glob import glob

os.environ['PYOPENGL_PLATFORM'] = 'egl'
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
neutral_bm_path = './imports/mdm/body_models/smplh/neutral/model.npz'
neutral_dmpl_path = './imports/mdm/body_models/dmpls/neutral/model.npz'
num_betas = 10
num_dmpls = 8 
bm = BodyModel(bm_fname=neutral_bm_path, num_betas=num_betas, num_dmpls=num_dmpls, dmpl_fname=neutral_dmpl_path).to(comp_device)
smplxmodel = smplx.create(model_path="imports/hmr4d/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).cpu()
trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


example_id = "000021"
l_idx1, l_idx2 = 5, 8
fid_r, fid_l = [8, 11], [7, 10]
face_joint_indx = [2, 1, 17, 16]
r_hip, l_hip = 2, 1
joints_num = 22
example_data_dir = "imports/mdm/dataset/HumanML3D/joints"

n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
kinematic_chain = t2m_kinematic_chain

example_data = np.load(os.path.join(example_data_dir, example_id + '.npy'))
example_data = example_data.reshape(len(example_data), -1, 3)
example_data = torch.from_numpy(example_data)
tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])


def amass_to_pose(src_path, save_path):
    bdata = np.load(src_path, allow_pickle=True)

    frame_num = bdata["poses"].shape[0]
    bdata_poses = bdata["poses"][:, 3 : 3 + 21*3]
    bdata_trans = bdata["trans"][:, :3]
    left_hand_pose = np.zeros((frame_num, 45))
    right_hand_pose = np.zeros((frame_num, 45))

    bdata_hand_poses = np.concatenate([left_hand_pose, right_hand_pose], axis=1)

    body_parms = {
            'root_orient': torch.Tensor(bdata["poses"][:, :3]).to(comp_device),
            'pose_body': torch.Tensor(bdata_poses).to(comp_device),
            'pose_hand': torch.Tensor(bdata_hand_poses).to(comp_device),
            'trans': torch.Tensor(bdata_trans).to(comp_device),
            'betas': torch.Tensor(np.repeat(bdata["betas"].reshape((1, 10)), frame_num, axis=0)[:, :num_betas]).to(comp_device),
        }
    
    with torch.no_grad():
        body = bm(**body_parms)

    pose_seq_np = body.Jtr.detach().cpu().numpy()
    pose_seq_np_n = np.dot(pose_seq_np, trans_matrix)

    np.save(save_path, pose_seq_np_n)


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def process_file(positions, feet_thre):
    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions[0][:, 1].min()
    positions[:, :, 1] -= floor_height

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    '''New ground truth positions'''
    global_positions = positions.copy()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    feet_l, feet_r = foot_detect(positions, feet_thre)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        r_rot = quat_params[:, 0].copy()
        '''Root Linear Velocity'''
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)
    data = root_data

    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)

    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def prepare_mdm_dataset(hmr_dir, pose_dir, new_joints_dir, new_joint_vecs_dir, mdm_dir, dataset, category):
    new_joint_vecs_pths = sorted(glob(f"{new_joint_vecs_dir}/{dataset}/{category}/*/*/*.npy"))
    new_joint_pths = sorted(glob(f"{new_joints_dir}/{dataset}/{category}/*/*/*.npy"))
    joints_pths = sorted(glob(f"{pose_dir}/{dataset}/{category}/*/*/*.npy"))
    
    print(SELECTED_INDICES)
    CATEGORY2TEXT_LIST = dict()
    all_file_names = []
    for ii, (joint_pth, new_joint_pth, new_joint_vec_pth) in enumerate(zip(joints_pths, new_joint_pths, new_joint_vecs_pths)):
        
        CATEGORY2TEXT_LIST[category] = [category]

        if SELECTED_INDICES.get(dataset, None) is not None and SELECTED_INDICES[dataset].get(category) is not None and ii not in SELECTED_INDICES[dataset][category]: continue
        left_indices = SELECTED_INDICES.get(f"left_{category}", None)
        right_indices = SELECTED_INDICES.get(f"right_{category}", None)

        train_data_joint_pth = f"{mdm_dir}/{dataset}/{category}/pose_data/{ii:06d}.npy"
        train_data_new_joint_pth = f"{mdm_dir}/{dataset}/{category}/new_joints/{ii:06d}.npy"
        train_data_new_joint_vec_pth = f"{mdm_dir}/{dataset}/{category}/new_joint_vecs/{ii:06d}.npy"
        train_data_text_pth = f"{mdm_dir}/{dataset}/{category}/texts/{ii:06d}.txt"

        all_file_names.append(f"{ii:06d}")

        os.makedirs("/".join(train_data_joint_pth.split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(train_data_new_joint_pth.split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(train_data_new_joint_vec_pth.split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(train_data_text_pth.split("/")[:-1]), exist_ok=True)


        os.system(f"cp '{joint_pth}' '{train_data_joint_pth}'")
        os.system(f"cp '{new_joint_pth}' '{train_data_new_joint_pth}'")
        os.system(f"cp '{new_joint_vec_pth}' '{train_data_new_joint_vec_pth}'")

        sampled_texts = random.sample(CATEGORY2TEXT_LIST[category], k=1)

        f = open(train_data_text_pth, "w")
        for sampled_text in sampled_texts:

            if left_indices is not None and ii in left_indices: sampled_text = f"left_{sampled_text}"
            if right_indices is not None and ii in right_indices: sampled_text = f"right_{sampled_text}"

            line = sampled_text
            caption = [x.text+'/'+x.pos_ for x in nlp(sampled_text)]
            line += "#"
            for cap in range(len(caption)):
                if cap != 0:
                    line += " "
                    line += caption[cap]
                else:
                    line += caption[cap]
            line += "#0.0#0.0\n"
            f.write(line)
        f.close()


    f = open(f"{mdm_dir}/{dataset}/{category}/all.txt", "w")
    for file_name in all_file_names:
        f.write(file_name + "\n")
    f.close()

    random.shuffle(all_file_names)

    f = open(f"{mdm_dir}/{dataset}/{category}/train.txt", "w")
    for file_name in all_file_names:
        f.write(file_name + "\n")
    f.close()

    f = open(f"{mdm_dir}/{dataset}/{category}/val.txt", "w")
    for file_name in all_file_names:
        f.write(file_name + "\n")
    f.close()

    f = open(f"{mdm_dir}/{dataset}/{category}/train_val.txt", "w")
    for file_name in all_file_names:
        f.write(file_name + "\n")
    f.close()

    f = open(f"{mdm_dir}/{dataset}/{category}/test.txt", "w")
    for file_name in all_file_names:
        f.write(file_name + "\n")
    f.close()

    mean_pth = "imports/mdm/dataset/HumanML3D/Mean.npy"
    train_data_mean_pth = f"{mdm_dir}/{dataset}/{category}/Mean.npy"
    std_pth = "imports/mdm/dataset/HumanML3D/Std.npy"
    train_data_std_pth = f"{mdm_dir}/{dataset}/{category}/Std.npy"

    os.system(f"cp '{mean_pth}' '{train_data_mean_pth}'")
    os.system(f"cp '{std_pth}' '{train_data_std_pth}'")


def process_mdm(
    dataset,
    category,
    hmr_dir,
    pose_dir,
    new_joints_dir,
    new_joint_vecs_dir,
    mdm_dir,
    skip_done
):
    if skip_done and os.path.exists(f"{mdm_dir}/{dataset}/{category}"): return

    hmr_pths = sorted(list(glob(f"{hmr_dir}/{dataset}/{category}/*/*/*.npz")))
    for hmr_pth in tqdm(hmr_pths):
        pose_data_save_path = hmr_pth.replace(hmr_dir, pose_dir).replace(".npz", ".npy")
        os.makedirs("/".join(pose_data_save_path.split("/")[:-1]), exist_ok=True)
        amass_to_pose(hmr_pth, pose_data_save_path)


    pose_data_pths = glob(f"{pose_dir}/{dataset}/{category}/*/*/*.npy")
    for pose_data_pth in tqdm(pose_data_pths):
        
        source_data = np.load(pose_data_pth)[:, :joints_num]
        data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
        rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)

        new_joint_path = pose_data_pth.replace(pose_dir, new_joints_dir)
        new_joint_vec_path = pose_data_pth.replace(pose_dir, new_joint_vecs_dir)

        os.makedirs("/".join(new_joint_path.split("/")[:-1]), exist_ok=True)
        os.makedirs("/".join(new_joint_vec_path.split("/")[:-1]), exist_ok=True)
        np.save(new_joint_path, rec_ric_data.squeeze().numpy())
        np.save(new_joint_vec_path, data)

    prepare_mdm_dataset(hmr_dir, pose_dir, new_joints_dir, new_joint_vecs_dir, mdm_dir, dataset, category)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--hmr_dir", type=str, default="results/generation/hmr")
    parser.add_argument("--pose_dir", type=str, default="results/david/pose_data")
    parser.add_argument("--new_joints_dir", type=str, default="results/david/new_joints")
    parser.add_argument("--new_joint_vecs_dir", type=str, default="results/david/new_joint_vecs")
    parser.add_argument("--mdm_dir", type=str, default="results/david/mdm")
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    process_mdm(
        dataset=args.dataset,
        category=args.category,
        hmr_dir=args.hmr_dir,
        pose_dir=args.pose_dir,
        new_joints_dir=args.new_joints_dir,
        new_joint_vecs_dir=args.new_joint_vecs_dir,
        mdm_dir=args.mdm_dir,
        skip_done=args.skip_done
    )