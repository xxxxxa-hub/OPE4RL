## Installation
For this project, we need to create two environments. One is for training and the other is for off-policy evaluation.

OPE4RL can be installed by cloning the repository as follows:
```
git clone https://github.com/xxxxxa-hub/OPE4RL.git
cd OPE4RL
bash setup.sh
```

## Basic Commands
In this repository, we mainly use Importance Sampling and Model-based method for off-policy evaluation. For Importance Sampling, we need to do behavior cloning on offline dataset. Similarly, we need to fit the dynamics model for Model-based method. For each task, the behavior or dynamics model should be pretrained once and can be used for evaluation for multiple times. The pretraining is done as follows:
```
CUDA_VISIBLE_DEVICES=0 /path/to/ope/bin/python /path/to/policy_eval/train.py --env_name Pendulum-replay --save_dir /home/featurize/checkpoints --target_policy_std 0.0 --seed 0 --algo iw --noise_scale 0.0 --lr 0.003 --lr_decay 1.0
```

With the pretrained model, we can start training with the following command:
```
/path/to/train/bin/python /path/to/d3rlpy/train.py --dataset Pendulum-replay --seed 1 --gpu cuda:0 --lr 1e-5 --batch_size 256 --temp 0.6 --ratio 1 --estimator_lr 0.003 --estimator_lr_decay 1.0 --n_epoch 200 --n_episodes 2 --algo iw --method baseline2 --upload --collect
```

`dataset`: The offline dataset we use for training.

`seed`: Random seed.

`gpu`: The device on which we run the experiment.

`lr`: Initial learning rate of both actor and critic network.

`ratio`: Ratio between the number of update steps of inner policy and that of outer policy.

`estimator_lr`: Initial learning rate of policy value estimator.

`estimator_lr_decay`: Decay rate of learning rate of estimator.

`n_epoch`: Number of training epochs.

`n_episodes`: Number of episodes to evaluate on-policy value.

`algo`: Algorithm of estimator training.

`method`: Paradigm of training. "baseline1" and "baseline2" are allowed in this project.

`upload`: Whether to upload the result to wandb for visualization.

`collect`: Whether to collect transitions during online evaluation.

