import cv2
import torch
import numpy as np
import argparse
from imports.hmr4d.utils.pylogger import Log
import hydra
from hydra import initialize_config_module, compose
from pathlib import Path
from pytorch3d.transforms import quaternion_to_matrix

from imports.hmr4d.configs import register_store_gvhmr
from imports.hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from imports.hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch

from imports.hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SLAMModel

from imports.hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from imports.hmr4d.utils.geo_transform import compute_cam_angvel
from imports.hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
from imports.hmr4d.utils.net_utils import detach_to_cpu, to_cuda
from imports.hmr4d.utils.smplx_utils import make_smplx
from imports.hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points
from tqdm import tqdm
from imports.hmr4d.utils.geo_transform import apply_T_on_points, compute_T_ayfz2ay
from einops import einsum, rearrange
from glob import glob
import os
import smplx
import pickle

CRF = 23  # 17 is lossless, every +6 halves the mp4 size

def parse_args_to_cfg(video, output_root, is_static_cam, verbose=False):
    # Input
    video_path = Path(video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")
    # Cfg
    with initialize_config_module(version_base="1.3", config_module=f"imports.hmr4d.configs"):
        overrides = [
            f"video_name={video_path.stem}",
            f"static_cam={is_static_cam}",
            f"verbose={verbose}",
            f"output_dir={video.replace('results', 'cache').replace('.mp4', '')}"
        ]

        # Allow to change output root
        if output_root is not None:
            overrides.append(f"output_root={output_root}")
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=overrides)

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get DPVO results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            length, width, height = get_video_lwh(cfg.video_path)
            K_fullimg = estimate_K(width, height)
            intrinsics = convert_K_to_K4(K_fullimg)
            print(intrinsics)
            slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
            bar = tqdm(total=length, desc="DPVO")
            while True:
                ret = slam.track()
                if ret:
                    bar.update()
                else:
                    break
            slam_results = slam.process()  # (L, 7), numpy
            torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        traj = torch.load(cfg.paths.slam)
        traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
        R_w2c = quaternion_to_matrix(traj_quat).mT
    K_fullimg = estimate_K(width, height).repeat(length, 1, 1)
    # K_fullimg = create_camera_sensor(width, height, 26)[2].repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features),
    }
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--video_dir", type=str, default="results/generation/videos")
    parser.add_argument("--hmr_save_dir", type=str, default="results/generation/hmr")
    parser.add_argument("--camera_motion_dir", type=str, default="results/generation/camera_motions")
    parser.add_argument("--static_cam", action="store_true", help="If true, skip DPVO")
    parser.add_argument("--verbose", action="store_true", help="If true, draw intermediate results")
    parser.add_argument("--device", default=0)
    parser.add_argument("--skip_done", action="store_true")
    args = parser.parse_args()

    data_path = "constants/smplx_handposes.npz"
    with np.load(data_path, allow_pickle=True) as data:
        hand_poses = data["hand_poses"].item()
        (left_hand_pose, right_hand_pose) = hand_poses["relaxed"]
        hand_pose_relaxed = np.concatenate( (left_hand_pose, right_hand_pose) ).reshape(1, -1)

    gpu = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    video_pths = glob(f"{args.video_dir}/{args.dataset}/{args.category}/*/*/*.mp4")
    for video_pth in video_pths:
        save_pth = video_pth.replace(args.video_dir, args.hmr_save_dir).replace(".mp4", ".npz")

        save_dir = "/".join(save_pth.split("/")[:-1])
        camera_motion_save_pth = video_pth.replace(args.video_dir, args.camera_motion_dir).replace(".mp4", ".pkl")
        camera_motion_save_dir = "/".join(camera_motion_save_pth.split("/")[:-1])
        
        if args.skip_done and os.path.exists(save_pth): continue
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(camera_motion_save_dir, exist_ok=True)

        cfg = parse_args_to_cfg(video_pth, output_root="cache", is_static_cam=args.static_cam, verbose=args.verbose)
        print(f"cfg: {cfg}")
        paths = cfg.paths
        Log.info(f"[GPU]: {torch.cuda.get_device_name()}")
        Log.info(f'[GPU]: {torch.cuda.get_device_properties("cuda")}')

        # ===== Preprocess and save to disk ===== #
        run_preprocess(cfg)
        data = load_data_dict(cfg)

        # ===== HMR4D ===== #
        if True:
            Log.info("[HMR4D] Predicting")
            model: DemoPL = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().to(gpu)
            tic = Log.sync_time()
            pred = model.predict(data, static_cam=cfg.static_cam)
            pred = detach_to_cpu(pred)
            data_time = data["length"] / 30
            Log.info(f"[HMR4D] Elapsed: {Log.sync_time() - tic:.2f}s for data-length={data_time:.1f}s")
            torch.save(pred, paths.hmr4d_results)

        pred_np = {}
        pred_np['smpl_params_global'] = {}
        pred_np['smpl_params_incam'] = {}
        pred_np['smpl_params_global']['body_pose'] = pred['smpl_params_global']['body_pose'].cpu().numpy()
        pred_np['smpl_params_global']['global_orient'] = pred['smpl_params_global']['global_orient'].cpu().numpy()
        pred_np['smpl_params_global']['transl'] = pred['smpl_params_global']['transl'].cpu().numpy()
        pred_np['smpl_params_global']['betas'] = pred['smpl_params_global']['betas'].cpu().numpy()
        
        pred_np['smpl_params_incam']['body_pose'] = pred['smpl_params_incam']['body_pose'].cpu().numpy()
        pred_np['smpl_params_incam']['global_orient'] = pred['smpl_params_incam']['global_orient'].cpu().numpy()
        pred_np['smpl_params_incam']['transl'] = pred['smpl_params_incam']['transl'].cpu().numpy()
        pred_np['smpl_params_incam']['betas'] = pred['smpl_params_incam']['betas'].cpu().numpy()


        ## PROCESS MOTION INCAM
        body_pose = pred_np["smpl_params_incam"]["body_pose"]# frame_num x 63
        betas = pred_np["smpl_params_incam"]["betas"] # frame_num x 10
        trans = pred_np["smpl_params_incam"]["transl"] # frame_num x 3
        global_orient = pred_np["smpl_params_incam"]["global_orient"] # frame_num x 3

        frame_num = body_pose.shape[0]

        animation_poses = np.zeros((frame_num, 55 * 3)) # frame_num x 165
        animation_poses[:, :1 * 3] = global_orient[:frame_num, :]
        animation_poses[:, 1 * 3 : 1 * 3 + 21 * 3] = body_pose[:frame_num, :]

        animation_betas = betas[0, :10] # frame_num x 10
        animation_trans = trans # frame_num x 3

        # No Hand Information !
        animation_poses[:, -30 * 3:] = hand_pose_relaxed

        motion_incam = dict(
            poses=animation_poses,
            betas=animation_betas,
            trans=animation_trans,
            mocap_frame_rate=30,
            gender="neutral"
        )

        ## PROCESS MOTION GLOBAL
        body_pose = pred_np["smpl_params_global"]["body_pose"]# frame_num x 63
        betas = pred_np["smpl_params_global"]["betas"] # frame_num x 10
        trans = pred_np["smpl_params_global"]["transl"] # frame_num x 3
        global_orient = pred_np["smpl_params_global"]["global_orient"] # frame_num x 3

        frame_num = body_pose.shape[0]

        animation_poses = np.zeros((frame_num, 55 * 3)) # frame_num x 165
        animation_poses[:, :1 * 3] = global_orient[:frame_num, :]
        animation_poses[:, 1 * 3 : 1 * 3 + 21 * 3] = body_pose[:frame_num, :]

        animation_betas = betas[0, :10] # frame_num x 10
        animation_trans = trans # frame_num x 3

        # No Hand Information !
        animation_poses[:, -30 * 3:] = hand_pose_relaxed

        motion_global = dict(
            poses=animation_poses,
            betas=animation_betas,
            trans=animation_trans,
            mocap_frame_rate=30,
            gender="neutral"
        )

        np.savez(save_pth, **motion_global)
        
        ## CAMERA GLOBAL
        motion_global_global_orient = motion_global["poses"][:, :3] # frame_num x 3
        motion_incam_global_orient = motion_incam["poses"][:, :3] # frame_num x 3

        frame_num = motion_global_global_orient.shape[0]

        motion_global_global_orient_MATRIX = torch.from_numpy(np.array([cv2.Rodrigues(motion_global_global_orient_frame)[0] for motion_global_global_orient_frame in motion_global_global_orient])).to(torch.float64)  # 3 x 3 x frame_num
        motion_global_global_orient_MATRIX_inverse = torch.from_numpy(np.array([np.linalg.inv(cv2.Rodrigues(motion_global_global_orient_frame)[0]) for motion_global_global_orient_frame in motion_global_global_orient])).to(torch.float64)  # 3 x 3 x frame_num
        motion_incam_global_orient_MATRIX = torch.from_numpy(np.array([cv2.Rodrigues(motion_incam_global_orient_frame)[0] for motion_incam_global_orient_frame in motion_incam_global_orient])).to(torch.float64)  # 3 x 3 x frame_num
        motion_incam_global_orient_MATRIX_inverse = torch.from_numpy(np.array([np.linalg.inv(cv2.Rodrigues(motion_incam_global_orient_frame)[0]) for motion_incam_global_orient_frame in motion_incam_global_orient])).to(torch.float64)  # 3 x 3 x frame_num

        motion_global_transl = torch.from_numpy(motion_global["trans"][:, :3].reshape((-1, 3, 1))).to(torch.float64) # frame_num x 3 x 1
        motion_incam_transl = torch.from_numpy(motion_incam["trans"][:, :3].reshape((-1, 3, 1))).to(torch.float64) # frame_num x 3 x 1
        
        smplxmodel = smplx.create(model_path="imports/hmr4d/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz", model_type="smplx", num_pca_comps=45).to(gpu)
        global_smplxmodel_output = smplxmodel(
            betas=torch.from_numpy(motion_global["betas"].reshape((1, 10))).repeat((frame_num, 1)).to(gpu).float(),
            global_orient=torch.from_numpy(motion_global["poses"][:, :3]).to(gpu).float(),
            body_pose=torch.from_numpy(motion_global["poses"][:, 3 : 3 + 21*3]).to(gpu).float(),
            left_hand_pose=torch.zeros((frame_num, 45)).to(gpu).float(),
            right_hand_pose=torch.zeros((frame_num, 45)).to(gpu).float(),
            transl=torch.from_numpy(motion_global["trans"][:, :3]).to(gpu).float(),
            expression=torch.zeros((frame_num, 10)).to(gpu).float(),
            jaw_pose=torch.zeros((frame_num, 3)).to(gpu).float(),
            leye_pose=torch.zeros((frame_num, 3)).to(gpu).float(),
            reye_pose=torch.zeros((frame_num, 3)).to(gpu).float(),
            return_verts=True,
            return_full_pose=True,
        )
        global_joints = global_smplxmodel_output.joints.to(torch.float64).cpu()
        global_pelvis_joints = global_joints[:, 0, :].cpu() # frame_num x 3

        incam_smplxmodel_output = smplxmodel(
            betas=torch.from_numpy(motion_incam["betas"].reshape((1, 10))).repeat((frame_num, 1)).to(gpu).float(),
            global_orient=torch.from_numpy(motion_incam["poses"][:, :3]).to(gpu).float(),
            body_pose=torch.from_numpy(motion_incam["poses"][:, 3 : 3 + 21*3]).to(gpu).float(),
            left_hand_pose=torch.zeros((frame_num, 45)).to(gpu).float(),
            right_hand_pose=torch.zeros((frame_num, 45)).to(gpu).float(),
            transl=torch.from_numpy(motion_incam["trans"][:, :3]).to(gpu).float(),
            expression=torch.zeros((frame_num, 10)).to(gpu).float(),
            jaw_pose=torch.zeros((frame_num, 3)).to(gpu).float(),
            leye_pose=torch.zeros((frame_num, 3)).to(gpu).float(),
            reye_pose=torch.zeros((frame_num, 3)).to(gpu).float(),
            return_verts=True,
            return_full_pose=True,
        )
        incam_joints = incam_smplxmodel_output.joints.to(torch.float64).cpu()
        incam_pelvis_joints = incam_joints[:, 0, :].cpu() # frame_num x 3

        motion_global_transl_ = global_pelvis_joints.reshape((-1, 3, 1)).to(torch.float64) # frame_num x 3 x 1
        motion_incam_transl_ = incam_pelvis_joints.reshape((-1, 3, 1)).to(torch.float64) # frame_num x 3 x 1

        R = torch.bmm(motion_incam_global_orient_MATRIX, motion_global_global_orient_MATRIX_inverse).to(torch.float64).transpose(1, 2) # frame_num x 3 x 3
        T = motion_global_transl_ - torch.bmm(R, motion_incam_transl_)

        with open(camera_motion_save_pth, 'wb') as handle:
            pickle.dump(dict(R=R.numpy(), t=T.to(torch.float64).numpy()), handle, protocol=pickle.HIGHEST_PROTOCOL)