# <p align="center"> DAViD: Modeling Dynamic Affordance of 3D Objects using Pre-trained Video Diffusion Models (ICCV 2025)</p>

## [Project Page](https://snuvclab.github.io/david/) &nbsp;|&nbsp; [Paper](https://arxiv.org/pdf/2501.08333) 

![demo.png](./assets/teaser.png)

This is the official code for the paper "DAViD: Modeling Dynamic Affordance of 3D Objects using Pre-trained Video Diffusion Models".

## Installation

To setup the environment for running ComA, please refer to the instructions provided <a href="INSTALL.md">here</a>.

## Quick Start

### 2D HOI Image Generation

To generate 2D HOI Images of given 3D object (in this case, barbell), use following command.

```shell
bash scripts/generate_2d_hoi_images.sh --dataset "ComAsset" --category "barbell" --device 0 --skip_done
```

### Image-to-Video

We leverage commercial image-to-video diffusion model [Kling AI](https://www.klingai.com/) to make 2D HOI videos from 2D HOI images.
Specifically, we use [imgur](https://imgur.com/) and [PiAPI](https://piapi.ai/docs) for uploading image and calling API for Kling AI. Check out `scripts/videos/get.sh`, `scripts/videos/post_i2v.sh` and setup your `X-API-key` of your PiAPI account. Also checkout `constants/videos.py` and setup your client id of your imgur account. Note that you need paid version of Kling AI for directly follow our setting.

```shell
CUDA_VISIBLE_DEVICES=0 python src/generation/generate_videos.py --dataset "ComAsset" --category "barbell" --skip_done
```

Otherwise, you can also use opensource image-to-video models such as [Wan2.1](https://github.com/Wan-Video/Wan2.1) but currently we haven't tested yet.

### 4D HOI Sample Generation

To generate 4D HOI Samples from the generated 2D HOI Images (of the given 3D object, frypan), use following command.

```shell
bash scripts/generate_4d_hoi_samples.sh --dataset "ComAsset" --category "barbell" --device 0 --skip_done
```

### Visualization

To visualize generated 4D HOI Samples, use following command

```shell
blenderproc debug src/visualization/visualize_4d_hoi_sample.py --dataset "ComAsset" --category "barbell" --idx 0
```

### Train LoRA for MDM

To train LoRA for MDM (of the given 3D object, barbell), use following command.

```shell
bash scripts/train_lora.sh --dataset "ComAsset" --category "barbell" --device 0
```

### Train Object Motion Diffusion Model

To train Object Motion Diffusion Model (of the given 3D object, barbell), use following command.

```shell
bash scripts/train_omdm.sh --dataset "ComAsset" --category "barbell" --device 0

```

### Sample Human Motion (Inference)

```shell
bash scripts/generate_human_motion.sh --max_seed 5 --dataset "ComAsset" --category "barbell" --device 0
```

### Sample Object Motion (Inference)

```shell
bash scripts/generate_object_motion.sh --dataset "ComAsset" --category "barbell" --device 0
```

## Regarding Code Release
- We are keep updating the code (including dataset and environment setup)!
- [2025/07/10] Initial skeleton code release!


## Citation
```bibtex
@misc{david,
      title={DAViD: Modeling Dynamic Affordance of 3D Objects using Pre-trained Video Diffusion Models}, 
      author={Hyeonwoo Kim and Sangwon Beak and Hanbyul Joo},
      year={2025},
      eprint={2501.08333},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.08333}, 
}
```