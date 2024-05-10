#!/bin/bash

set -e

# Training
cd d3rlpy
conda create -n train python=3.8
source activate train
pip install d4rl==1.1
pip install -e .
conda deactivate
cd ..

# OPE
conda create -n ope python=3.8
source activate ope
cd policy_eval
pip install tensorflow==2.6.0
pip install d4rl==1.1
conda install cudatoolkit cudnn
pip install -e .
pip install "cython<3"
pip install patchelf==0.17.2
pip uninstall pybullet
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
    conda install -c conda-forge libstdcxx-ng=12
    conda install -c conda-forge glew
    conda install -c conda-forge mesalib
    conda install -c menpo glfw3
    export CPATH=$CONDA_PREFIX/include
    echo "CPATH=\$CONDA_PREFIX/include" >> ~/.bashrc
fi
conda deactivate
cd ..
