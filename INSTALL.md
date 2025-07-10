# Installation
DAViD has been developed and tested on Ubuntu 20.04 with an NVIDIA GeForce RTX A6000 GPU device. To get started, follow the installation instructions below.

## Environment Setup

```shell
# install ubuntu dependencies
apt-get install libxrender1 libxi6 libglfw3-dev libgles2-mesa-dev curl
apt install -y libxkbcommon-x11-0 libsm6 libxext6
apt install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get install gcc-11 g++-11
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11
apt install ffmpeg

# make environment
conda create -n david python=3.10
conda activate david

# install PyTorch
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# install PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

# install detectron2
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder detectron2==0.6+2a420edpt2.4.0cu121

# install visualizers
pip install blenderproc==2.6.2
blenderproc pip install tqdm
[blenderproc pip path] install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
[blenderproc pip path] install -U fvcore
[blenderproc pip path] install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu113_pyt1121/download.html
[blenderproc pip path] install certifi
export SSL_CERT_DIR=/etc/ssl/certs/

# install DPVO
cd imports/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
pip install torch-scatter -f "https://data.pyg.org/whl/torch-2.4.0+cu121.html"
pip install numba==0.57.0 pypose
export CUDA_HOME=/usr/local/cuda-12.1/
export PATH=$PATH:/usr/local/cuda-12.1/bin/
pip install -e .
cd ..

# install depthpro
cd imports/ml-depth-pro
pip install -e .
pip install timm==1.0.3
source get_pretrained_models.sh
cd ..

# install mdm
cd imports/mdm/checkpoints
pip install gdown
gdown --id 1cfadR1eZ116TIdXK7qDX1RugAerEiJXr # for downloading mdm checkpoints 
unzip humanml_enc_512_50steps.zip 
rm humanml_enc_512_50steps.zip
cd ..
bash prepare/download_smpl_files.sh
bash prepare/download_glove.sh
bash prepare/download_t2m_evaluators.sh


# install other dependencies
pip install transformers accelerate sentencepiece smplx

pip install git+https://github.com/huggingface/diffusers
pip install ftfy moviepy==1.0.3 hydra-zen hydra_colorlog ffmpeg lightning==2.3.0 ultralytics==8.2.42 av imageio[pyav] pandas lora-pytorch blobfile wandb spacy ipdb tensorboardX natsort open3d
python -m spacy download en_core_web_sm
pip install git+https://github.com/openai/CLIP.git

pip install -e .
pip install diffusers==0.20.2 accelerate safetensors transformers
pip install numpy==1.23.1 loguru numba filterpy flatten-dict smplx trimesh==3.23.5 jpeg4py chumpy easydict pickle5 torchgeometry networkx==2.8 pysdf mayavi PyQt5==5.14.2 jupyter yq tqdm supervision Pillow==9.5.0 open3d plyfile openai configer
pip install pyopengl==3.1.4 pyrender==0.1.45
pip install segment-anything
```

## Prepare Datasets

We use [ComAsset](https://huggingface.co/datasets/SShowbiz/ComAsset) as our main dataset. Please download the dataset and place below the folder `data` as follows. You can also consider other datasets such as [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/), [InterCap](https://intercap.is.tue.mpg.de/), [FullBodyManip](https://github.com/lijiaman/omomo_release), [SAPIEN](https://sapien.ucsd.edu/). Please refer to our `utils/dataset.py` for the dataset placement.

```
data
└── ComAsset
    ├── accordion # object category
    │   └── wx75e99elm1yhyfxz1efg60luadp95sl # object id
    │       ├── images # folder for texture files
    │       ├── model.obj
    │       └── model.mtl
    ├── axe
    ├── ...
    └── watering can
```