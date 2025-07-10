import argparse
import os
from imports.mdm.visualize import vis_utils
import shutil
from tqdm import tqdm
from glob import glob
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="frypan")
    parser.add_argument("--human_motion_dir", type=str, default="results/inference/human_motion")
    parser.add_argument("--frame_num", type=int, default=None)

    parser.add_argument("--cuda", type=bool, default=True, help='')
    parser.add_argument("--device", type=int, default=0, help='')
    parser.add_argument("--sample", type=int, default=0, help='')

    parser.add_argument("--skip_done", action="store_true")

    params = parser.parse_args()

    data_path = "constants/smplx_handposes.npz"
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(1, -1)


    mp4_pths = glob(f"{params.human_motion_dir}/{params.dataset}/{params.category}/*/*/*.mp4")
    for input_path in mp4_pths:
        assert input_path.endswith('.mp4')
        
        parsed_name = os.path.basename(input_path).replace('.mp4', '').replace('sample', '').replace('rep', '')
        sample_i, rep_i = [int(e) for e in parsed_name.split('_')]
        npy_path = os.path.join(os.path.dirname(input_path), 'results.npy')
        out_npy_path = input_path.replace('.mp4', '_smpl_params.npy')
        out_npz_path = out_npy_path.replace(".npy", ".npz")

        if params.skip_done and os.path.exists(out_npy_path): continue
        
        assert os.path.exists(npy_path)
        results_dir = input_path.replace('.mp4', '_obj')

        if os.path.exists(results_dir):
            shutil.rmtree(results_dir)
        os.makedirs(results_dir)
        os.makedirs(os.path.join(results_dir, "loc"))

        npy2obj = vis_utils.npy2obj(npy_path, sample_i, rep_i, device=params.device, cuda=params.cuda)

        print('Saving SMPL params to [{}]'.format(os.path.abspath(out_npy_path)))
        npy_data = npy2obj.save_npy(out_npy_path)

        animation_info = npy_data
        body_pose = animation_info["opt_dict"]["pose"][:, 1 * 3: 1 * 3 + 21 * 3].detach().cpu().numpy() # frame_num x 63
        betas = animation_info["opt_dict"]["betas"].detach().cpu().numpy() # frame_num x 10

        trans = animation_info["root_translation"].T # frame_num x 3
        global_orient = animation_info["opt_dict"]["pose"][:, : 1 * 3].detach().cpu().numpy() # frame_num x 3

        frame_num = params.frame_num if params.frame_num is not None else body_pose.shape[0]
        
        animation_poses = np.zeros((frame_num, 55 * 3)) # frame_num x 155
        animation_poses[:, :1 * 3] = global_orient[:frame_num, :]
        animation_poses[:, 1 * 3 : 1 * 3 + 21 * 3] = body_pose[:frame_num, :]

        # TODO: change it!
        animation_poses[:, 15 * 3 : 1 * 3 + 15 * 3] = np.zeros((frame_num, 3)) # cannot know head information
        animation_poses[:, 12 * 3 : 1 * 3 + 12 * 3] = np.zeros((frame_num, 3)) # cannot know head information

        animation_betas = betas[0, :10] # frame_num x 10
        animation_trans = trans # frame_num x 3

        # No Hand Information !
        animation_poses[:, -30 * 3:] = hand_pose_relaxed

        to_save = dict(
            poses=animation_poses,
            betas=animation_betas,
            trans=animation_trans,
            mocap_frame_rate=30,
            gender="neutral"
        )

        np.savez(out_npz_path, **to_save)