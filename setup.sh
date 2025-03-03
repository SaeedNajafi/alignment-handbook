#!/bin/bash

set -e

ENV_NAME=$1

conda create -n ${ENV_NAME} python=3.11

eval "$(conda shell.bash hook)"

conda activate ${ENV_NAME}

# echo "Installing cxx-compiler and gxx and gcc."
# conda install -c conda-forge cxx-compiler gxx gcc level-zero -y
# conda install libevent -y

# echo "Installing git."
# conda install -c anaconda git -y

# echo "Installing git-lfs."
# conda install -c conda-forge git-lfs -y

# echo "Installing rust."
# conda install -c conda-forge rust -y

# echo "Installing cuda drivers."
# conda install -c nvidia/label/cuda-12.4.1 cuda cuda-nvcc cuda-toolkit -y
# conda install -c conda-forge cudnn -y

echo "Upgrade pip."
pip3 install --upgrade pip

echo "Install torch."
pip3 install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# export CUDA_HOME=$CONDA_PREFIX
# export NCCL_HOME=$CONDA_PREFIX
# export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

echo "Fix sqlite"
conda upgrade sqlite -y

pip3 install --no-cache-dir flash-attn --no-build-isolation

pip3 install vllm ray llvmlite vllm-flash-attn

# pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4
