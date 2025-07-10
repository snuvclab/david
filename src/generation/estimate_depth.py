import argparse
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
import os
import depth_pro
import numpy as np
from PIL import Image
from tqdm import tqdm

INITIAL_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

def match_depth(
    dataset,
    category,
    object_vertices_dir,
    image_dir,
    depth_dir,
    depthmap_dir,
    skip_done
):
    model, transform = depth_pro.create_model_and_transforms()
    model.eval().to('cuda')

    object_vertices_pths = glob(f"{object_vertices_dir}/{dataset}/{category}/*/*/*.pkl")

    for object_vertices_pth in tqdm(object_vertices_pths):
        image_pth = "/".join(object_vertices_pth.replace(object_vertices_dir, image_dir).split("/")[:-1]) + ".png"
        depthmap_pth = object_vertices_pth.replace(object_vertices_dir, depthmap_dir).replace(".pkl", ".png")
        depth_pth = object_vertices_pth.replace(object_vertices_dir, depth_dir)
        depthmap_save_dir = "/".join(depthmap_pth.split("/")[:-1])
        depth_save_dir = "/".join(depth_pth.split("/")[:-1])

        if skip_done and os.path.exists(depthmap_pth) and os.path.exists(depth_pth): continue

        os.makedirs(depthmap_save_dir, exist_ok=True)
        os.makedirs(depth_save_dir, exist_ok=True)
        
        image, _, f_px = depth_pro.load_rgb(image_pth)
        image = transform(image).to('cuda')
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy().squeeze()  # Depth in [m].

        inverse_depth = 1 / depth
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )
        cmap = plt.get_cmap("turbo")
        color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(color_depth).save(
            depthmap_pth, format="JPEG", quality=90
        )
        
        with open(depth_pth, "wb") as handle:
            pickle.dump(
                dict(depth=depth),
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")

    parser.add_argument("--object_vertices_dir", type=str, default="results/generation/object_vertices")
    parser.add_argument("--image_dir", type=str, default="results/generation/images")
    parser.add_argument("--depth_dir", type=str, default="results/generation/depth")
    parser.add_argument("--depthmap_dir", type=str, default="results/generation/depthmap")
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    match_depth(
        dataset=args.dataset,
        category=args.category,
        object_vertices_dir=args.object_vertices_dir,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        depthmap_dir=args.depthmap_dir,
        skip_done=args.skip_done
    )