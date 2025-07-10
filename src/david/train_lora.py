# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from imports.mdm.utils.fixseed import fixseed
# from imports.mdm.utils.parser_util import train_args
from imports.mdm.utils.parser_util import add_base_options, add_data_options, add_model_options, add_diffusion_options, add_training_options
from imports.mdm.utils import dist_util
from imports.mdm.train.training_loop import TrainLoop
from imports.mdm.data_loaders.get_data import get_dataset_loader
from imports.mdm.utils.model_util import create_model_and_diffusion
from imports.mdm.train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation
import argparse

def train_lora(
    t_args,
    david_dataset,
    category,
    checkpont_save_dir,
    skip_done
):
    t_args.is_train = True
    t_args.save_dir = f"{checkpont_save_dir}/{david_dataset}/{category}"
    
    fixseed(args.seed)
    train_platform_type = eval(t_args.train_platform_type)
    train_platform = train_platform_type(t_args.save_dir)
    train_platform.report_args(t_args, name='Args')

    if t_args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(t_args.save_dir) and not t_args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(t_args.save_dir))
    elif not os.path.exists(t_args.save_dir):
        os.makedirs(t_args.save_dir)
    args_path = os.path.join(t_args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(t_args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(t_args.device)

    print("creating data loader...")
    data = get_dataset_loader(name=t_args.dataset, batch_size=t_args.batch_size, num_frames=t_args.num_frames, david_dataset=david_dataset, david_category=category)


    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(t_args, data)
    model.to(dist_util.dev())
    model.rot2xyz.smpl_model.eval()

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    TrainLoop(t_args, train_platform, model, diffusion, data).run_loop()
    train_platform.close()


def train_args():
    
    return parser.parse_args()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)

    group = parser.add_argument_group('david')
    group.add_argument("--david_dataset", type=str, default="ComAsset")
    group.add_argument("--category", type=str, default="barbell")
    group.add_argument("--checkpont_save_dir", type=str, default="results/david/lora")
    group.add_argument("--skip_done", action="store_true")

    args = parser.parse_args()

    train_lora(
        t_args=args,
        david_dataset=args.david_dataset,
        category=args.category,
        checkpont_save_dir=args.checkpont_save_dir,
        skip_done=args.skip_done
    )