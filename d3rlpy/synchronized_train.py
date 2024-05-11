import argparse
import pdb
import numpy as np
import d3rlpy
import pickle
import d4rl
import d4rl.gym_mujoco
import gym
import torch
import os
from d3rlpy.dataset import ReplayBuffer_, D4rlDataset, get_cartpole, get_pendulum, infinite_loader
from d3rlpy.algos.qlearning.model_sac import Model
from torch.utils.data import DataLoader
from utils import SynchronizedExperiment
from torch.multiprocessing import Process, set_start_method

# 0, 1, 1234
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="walker2d-medium-v0")
    parser.add_argument("--method", type=str, default="baseline2") # "new" or "baseline"
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="cuda:1")
    
    # Hyper-parameter for offline training
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    
    # Hyper-parameter for pre-trained estimator
    parser.add_argument("--estimator_lr", type=float, default=0.003)
    
    # Environment path
    parser.add_argument("--save_dir", type=str, default="checkpoints_v11")
    parser.add_argument("--python_file", type=str, default=os.path.join(os.path.dirname(os.environ["CONDA_PREFIX"]), "ope", "bin", "python"))
    parser.add_argument("--eval_file", type=str, default="../policy_eval/eval.py")
    
    # Configuration of training
    parser.add_argument("--collect_epoch", type=int, default=100)
    parser.add_argument("--n_epoch", type=int, default=200)
    parser.add_argument("--n_episodes", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--algo", type=str, default="mb") # "iw" or "mb"
    
    parser.add_argument('--upload', dest='upload', action='store_true', help='Enable upload')
    parser.add_argument('--no-upload', dest='upload', action='store_false', help='Disable upload')
    parser.add_argument('--collect', dest='collect', action='store_true', help='Enable collect')
    parser.add_argument('--no-collect', dest='collect', action='store_false', help='Disable collect')
    parser.set_defaults(upload=True)
    parser.set_defaults(collect=True)

    args = parser.parse_args()

    if "Pendulum" in args.dataset:
        env = gym.make("Pendulum-v1")
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = get_pendulum(dataset_type=args.dataset.split("-")[1])
    elif "CartPole" in args.dataset:
        env = gym.make("CartPole-v1")
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = get_cartpole(dataset_type=args.dataset.split("-")[1])
    else:
        env = gym.make(args.dataset)
        d3rlpy.seed(args.seed)
        d3rlpy.envs.seed_env(env, args.seed)
        d4rl_dataset = env.get_dataset()


    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])

    if "medium-v0" in args.dataset:
        conservative_weight = 10.0
    else:
        conservative_weight = 5.0

    experiment = SynchronizedExperiment(lr_range=[1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3], # , 3e-4, 5e-4, 7e-4, 1e-3
                                        encoder=encoder,
                                        batch_size=args.batch_size,
                                        conservative_weight=conservative_weight,
                                        device=args.gpu)

    if args.method == "baseline1":
        experiment.fit(
            method=args.method,
            dataset=d4rl_dataset,
            n_epoch=args.n_epoch,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995, max_steps=args.max_steps)},
            python_file=args.python_file,
            eval_file=args.eval_file,
            save_dir=args.save_dir,
            seed = args.seed,
            env_name = args.dataset,
            collect_epoch = args.collect_epoch,
            estimator_lr = args.estimator_lr,
            algo = args.algo,
            upload = args.upload,
            collect = args.collect
        )
    elif args.method == "baseline2":
        experiment.fit(
            method=args.method,
            dataset=d4rl_dataset,
            n_epoch=args.n_epoch,
            evaluators={"environment": d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995, max_steps=args.max_steps)},
            python_file=args.python_file,
            eval_file=args.eval_file,
            save_dir=args.save_dir,
            seed = args.seed,
            env_name = args.dataset,
            collect_epoch = args.collect_epoch,
            estimator_lr = args.estimator_lr,
            algo = args.algo,
            upload = args.upload,
            collect = args.collect
        )
    


if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()