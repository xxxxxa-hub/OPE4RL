#!/bin/sh

conda create -n train python=3.8
conda create -n ope python=3.8
source activate train && cd ~/OPE4RL/d3rlpy && bash setup.sh
source activate ope && cd ~/OPE4RL/policy_eval && bash setup.sh

