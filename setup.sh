#!/bin/bash

set -e

# configure mujoco
if [ ! -d ~/.mujoco/mujoco210 ]; then
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
    tar -xzf mujoco210-linux-x86_64.tar.gz
    mkdir -p ~/.mujoco
    mv mujoco210 ~/.mujoco/
    rm mujoco210-linux-x86_64.tar.gz
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin
    echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin" >> ~/.bashrc
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
    echo "LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/nvidia" >> ~/.bashrc
    conda install -c conda-forge libstdcxx-ng=12 glew mesalib
    conda install -c menpo glfw3
    export CPATH=$CONDA_PREFIX/include
    echo "CPATH=\$CONDA_PREFIX/include" >> ~/.bashrc
    pip install patchelf==0.17.2
fi

# get d4rl
cd ~
git clone https://github.com/rail-berkeley/d4rl.git


# env: train
conda create -n clean_train python=3.8
source activate clean_train
cd d3rlpy
pip install -e .
# install d4rl for env: train
cd ~/d4rl
pip install -e . --no-deps
conda deactivate

# env: ope
conda create -n clean_ope python=3.8
source activate clean_ope
pip install tensorflow==2.6.0
conda install cudatoolkit cudnn
cd ~/OPE-latest/policy_eval
pip install -e .
# install d4rl for env: ope
cd ~/d4rl
pip install -e . --no-deps
conda deactivate
