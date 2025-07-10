import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from glob import glob
import pickle
import cv2
import os
from constants.config import WIDTH, HEIGHT

FOCAL_LENGTH = (WIDTH ** 2 + HEIGHT ** 2) ** 0.5
X_INVERSE_MATRIX = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
CAMERA_MATRIX = np.array([[FOCAL_LENGTH, 0, WIDTH / 2], [0, FOCAL_LENGTH, HEIGHT / 2], [0, 0, 1]]) 
DIST_COEFFS = np.zeros((4, 1))

DEVICE = 'cuda'
def pnp(
    dataset,
    category,
    track_dir,
    camera_motion_dir,
    pnp_dir,
    skip_done
):
    track_pths = glob(f"{track_dir}/{dataset}/{category}/*/*/*.pkl")

    for track_pth in track_pths:
        camera_motion_pth = track_pth.replace(track_dir, camera_motion_dir)
        pnp_pth = track_pth.replace(track_dir, pnp_dir)
        pnp_save_dir = "/".join(pnp_pth.split("/")[:-1])
        
        if skip_done and os.path.exists(pnp_pth): continue
        os.makedirs(pnp_save_dir, exist_ok=True)

        with open(track_pth, "rb") as handle:
            track_data = pickle.load(handle)

        with open(camera_motion_pth, "rb") as handle:
            camera_motion_data = pickle.load(handle)
        
        camera_motion_R = np.array(camera_motion_data["R"]) # frame_num x 3 x 3
        camera_motion_t = np.array(camera_motion_data["t"]).reshape((-1, 3, 1)) # frame_num x 3 x 1

        sampled_vertices_3D = np.array([sampled_vertex["vertices_3D"] for sampled_vertex in track_data["sampled_vertices"]]) # sampled_vertices_num x 3
        pred_tracks = np.array([  [[track[0], track[1]] for track in frame_track] for frame_track in track_data["pred_tracks"]]) # frame_num x sampled_vertices_num x 2
        frame_num = camera_motion_R.shape[0]
        
        if len(sampled_vertices_3D) < 4: continue
        pnp_save = []
        for idx in range(frame_num):
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objectPoints=sampled_vertices_3D, imagePoints=pred_tracks[idx], cameraMatrix=CAMERA_MATRIX, distCoeffs=None, iterationsCount=10000, reprojectionError=8.0, flags=cv2.SOLVEPNP_EPNP)
            pnp_cam_R = cv2.Rodrigues(rvecs)[0]
            pnp_cam_t = tvecs.reshape((3, 1))

            inferred_cam_R = np.linalg.inv(pnp_cam_R) @ X_INVERSE_MATRIX
            inferred_cam_t = - np.linalg.inv(pnp_cam_R) @ pnp_cam_t

            real_camera_R = camera_motion_R[idx]
            real_camera_t = camera_motion_t[idx]

            pnp_result = dict(
                inferred_cam_R=inferred_cam_R,
                inferred_cam_t=inferred_cam_t,
                real_camera_R=real_camera_R,
                real_camera_t=real_camera_t
            )
            pnp_save.append(pnp_result)

        with open(pnp_pth, "wb") as handle:
            pickle.dump(
                pnp_save,
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--track_dir", type=str, default="results/generation/track")
    parser.add_argument("--camera_motion_dir", type=str, default="results/generation/camera_motions_blender")
    parser.add_argument("--pnp_dir", type=str, default="results/generation/pnp")
    parser.add_argument("--skip_done", action="store_true")


    args = parser.parse_args()
    pnp(
        dataset=args.dataset,
        category=args.category,
        track_dir=args.track_dir,
        camera_motion_dir=args.camera_motion_dir,
        pnp_dir=args.pnp_dir,
        skip_done=args.skip_done
    )