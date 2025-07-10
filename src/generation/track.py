# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import argparse
import numpy as np

from PIL import Image
from imports.cotracker.utils.visualizer import Visualizer, read_video_from_path
from imports.cotracker.predictor import CoTrackerPredictor
import pickle
import random
from glob import glob
from constants.config import WIDTH, HEIGHT

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

def track(
    dataset,
    category,
    video_dir,
    mask_path,
    checkpoint,
    grid_size,
    grid_query_frame,
    backward_tracking,
    use_v2_model,
    offline,
    object_vertices_dir,
    track_dir,
    skip_done,
):
    video_pths = sorted(glob(f"{args.video_dir}/{dataset}/{category}/*/*/*.mp4"))

    for video_pth in video_pths:
        vertices_pth = video_pth.replace(video_dir, object_vertices_dir).replace(".mp4", ".pkl")
        track_pth = vertices_pth.replace(object_vertices_dir, track_dir)
        track_save_dir = "/".join(track_pth.split("/")[:-1])
        track_video_log_dir = video_pth.replace(video_dir, "tracklog").replace(".mp4", "")

        os.makedirs(track_save_dir, exist_ok=True)
        os.makedirs(track_video_log_dir, exist_ok=True)

        if skip_done and os.path.exists(track_pth): continue

        # load the input video frame by frame
        video = read_video_from_path(video_pth)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        if args.checkpoint is not None:
            if args.use_v2_model:
                model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
            else:
                if args.offline:
                    window_len = 60
                else:
                    window_len = 16
                model = CoTrackerPredictor(
                    checkpoint=args.checkpoint,
                    v2=args.use_v2_model,
                    offline=args.offline,
                    window_len=window_len,
                )
        else:
            model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

        with open(vertices_pth, "rb") as handle:
            vertices = pickle.load(handle)
        
        sampled_indices = random.sample(range(len(vertices)), min(len(vertices), 2000))
        sampled_vertices = [vertices[index] for index in sampled_indices]

        vertices_ = np.array([[0.0, vertex["vertices_2D"][0], HEIGHT - (vertex["vertices_2D"][1])] for vertex in sampled_vertices])
        queries = torch.from_numpy(vertices_).unsqueeze(0).to(DEFAULT_DEVICE).float()

        model = model.to(DEFAULT_DEVICE)
        video = video.to(DEFAULT_DEVICE)

        try:
            pred_tracks, pred_visibility = model(
                video,
                queries=queries,
                backward_tracking=args.backward_tracking,
            )
        except:
            continue

        true_mask = pred_visibility.squeeze(0).sum(dim=0) > int(pred_visibility.shape[1] // 10 * 7)
        pred_visibility = pred_visibility[:, :, true_mask.squeeze()]
        pred_tracks = pred_tracks[:, :, true_mask.squeeze(), :]
        sampled_vertices = [sampled_vertices[idx] for idx, mask in enumerate(true_mask.squeeze()) if mask]

        with open(track_pth, "wb") as handle:
            pickle.dump(
                dict(
                    sampled_vertices=sampled_vertices,
                    pred_tracks=pred_tracks.squeeze(0).cpu().numpy()
                ),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        # save a video with predicted tracks
        vis = Visualizer(save_dir=track_video_log_dir, pad_value=120, linewidth=3)
        vis.visualize(
            video,
            pred_tracks,
            pred_visibility,
            query_frame=0 if args.backward_tracking else args.grid_query_frame,
            save_video=True
        )
     


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--video_dir", default="results/generation/videos", help="path to a video")

    parser.add_argument("--mask_path", default="./assets/apple_mask.png", help="path to a segmentation mask",)
    parser.add_argument("--checkpoint", default=None, help="CoTracker model parameters",)
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument("--grid_query_frame", type=int, default=0, help="Compute dense and grid tracks starting from this frame",)
    parser.add_argument("--backward_tracking", action="store_true", help="Compute tracks in both directions, not only forward",)
    parser.add_argument("--use_v2_model", action="store_true", help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",)
    parser.add_argument("--offline", action="store_true", help="Pass it if you would like to use the offline model, in case of online don't pass it",)
    parser.add_argument("--object_vertices_dir", default="results/generation/object_vertices")
    parser.add_argument("--track_dir", default="results/generation/track")
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()

    track(
        dataset=args.dataset,
        category=args.category,
        video_dir=args.video_dir,
        mask_path=args.mask_path,
        checkpoint=args.checkpoint,
        grid_size=args.grid_size,
        grid_query_frame=args.grid_query_frame,
        backward_tracking=args.backward_tracking,
        use_v2_model=args.use_v2_model,
        offline=args.offline,
        object_vertices_dir=args.object_vertices_dir,
        track_dir=args.track_dir,
        skip_done=args.skip_done,
    )