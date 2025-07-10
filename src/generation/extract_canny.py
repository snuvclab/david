from glob import glob
from tqdm import tqdm
import cv2
import argparse
import os


def extract_canny(
    dataset,
    category,
    object_image_save_dir,
    object_canny_save_dir,
    skip_done
):
    if dataset is not None and category is not None:
        object_image_pths = glob(f"{object_image_save_dir}/{dataset}/{category}/*.png")
    else: object_image_pths = glob(f"{object_image_save_dir}/*/*.png")

    for object_image_pth in tqdm(object_image_pths):
        save_path = object_image_pth.replace(object_image_save_dir, object_canny_save_dir)
        save_dir = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)

        if skip_done and os.path.exists(save_path): continue
        
        object_image_np = cv2.imread(object_image_pth, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(object_image_np, 30, 40)

        cv2.imwrite(save_path, edges)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## save directories
    parser.add_argument("--object_image_save_dir", type=str, default="results/generation/asset_renders")
    parser.add_argument("--object_canny_save_dir", type=str, default="results/generation/canny_edges")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    extract_canny(
        dataset=args.dataset,
        category=args.category,
        object_image_save_dir=args.object_image_save_dir,
        object_canny_save_dir=args.object_canny_save_dir,
        skip_done=args.skip_done
    )
