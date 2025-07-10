import torch
from diffusers.utils import load_image
from diffusers.models.controlnet_flux import FluxControlNetModel
from constants.config import CATEGORY2GENERATIONCONFIG
from utils.pipeline_flux_controlnet import FluxControlNetPipeline

from PIL import Image
from glob import glob
import argparse
import os
from tqdm import tqdm


def generate_images(
    dataset,
    category,
    object_canny_save_dir,
    image_save_dir,
    device,
    skip_done
):  
    base_model = 'black-forest-labs/FLUX.1-dev'
    controlnet_model = 'InstantX/FLUX.1-dev-Controlnet-Canny'
    controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
    pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
    pipe.to(f"cuda:{device}")
    
    canny_image_pths = list(sorted(glob(f"{object_canny_save_dir}/{dataset}/{category}/*.png")))
    for canny_image_pth in tqdm(canny_image_pths):
        for seed in range(0, 500):
            generator = torch.Generator()
            generator.manual_seed(seed)
            
            save_dir = canny_image_pth.replace(object_canny_save_dir, image_save_dir).split(".")[0]
            save_pth = f"{save_dir}/{seed:05d}.png"

            os.makedirs(save_dir, exist_ok=True)

            if skip_done and os.path.exists(save_pth): continue

            control_image_canny = Image.open(canny_image_pth).convert("RGB")
            control_mode_canny = 0

            width, height = control_image_canny.size

            image = pipe(
                prompt = CATEGORY2GENERATIONCONFIG[category]["prompt"], 
                control_image=control_image_canny,
                control_mode=control_mode_canny,
                width=width,
                height=height,
                controlnet_conditioning_scale=0.6,
                num_inference_steps=28, 
                guidance_scale=3.5,
                generator=generator,
            ).images[0]

            image.save(save_pth)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## save directories
    parser.add_argument("--dataset", type=str, default="ComAsset")
    parser.add_argument("--category", type=str, default="barbell")
    parser.add_argument("--object_canny_save_dir", type=str, default="results/generation/canny_edges")
    parser.add_argument("--image_save_dir", type=str, default="results/generation/images")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()
    generate_images(
        dataset=args.dataset,
        category=args.category,
        object_canny_save_dir=args.object_canny_save_dir,
        image_save_dir=args.image_save_dir,
        device=args.device,
        skip_done=args.skip_done
    )