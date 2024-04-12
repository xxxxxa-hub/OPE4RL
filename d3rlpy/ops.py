import argparse
import pandas as pd
import numpy as np
import os
import pdb
import torch
import d3rlpy
import gym
import d4rl
from d3rlpy.utils import run
from save import process_baseline1, process_baseline2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="halfcheetah-medium-replay-v0")
    parser.add_argument("--algo", type=str, default="mb")
    parser.add_argument("--method", type=str, default="baseline1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=120)
    args = parser.parse_args()
    
    method = args.method
    save_dir = "/home/xiaoan/checkpoints_v7"
    dataset = args.dataset
    python_file = "/home/xiaoan/miniconda3/envs/ope/bin/python"
    eval_file = "/home/xiaoan/OPE4RL/policy_eval/eval.py"
    lr = 0.003
    algo = args.algo
    seed = args.seed

    env = gym.make("halfcheetah-medium-replay-v0")
    d3rlpy.seed(0)
    d3rlpy.envs.seed_env(env, 0)


    if method == "baseline1":
        dir_path = "{}/{}/{}".format(save_dir, method, dataset)
    elif method == "baseline2":
        dir_path = "{}/{}/{}/{}".format(save_dir, method, dataset, algo)

    hp_list = os.listdir(dir_path)

    # if method == "baseline1":
    #     process_baseline1(save_dir=save_dir, dataset=dataset, python_file=python_file,
    #                      eval_file=eval_file, lr=lr, algo=algo)
    # elif method == "baseline2":
    #     process_baseline2(save_dir=save_dir, dataset=dataset, python_file=python_file,
    #                      eval_file=eval_file, lr=lr, algo=algo)

    estimate_list = []

    for hp in hp_list:
        seed_dir_path = os.path.join(dir_path, hp ,"seed{}".format(seed))
        estiamte = pd.read_csv(os.path.join(seed_dir_path, "ope_{}.csv".format(args.epoch))).iloc[0,0]
        estimate_list.append(estiamte)

    print(estimate_list)
    print(len(estimate_list))
    index = sorted(range(len(estimate_list)), key=lambda i: estimate_list[i], reverse=True)[0]
    print(index)
    print(max(estimate_list))
    print(hp_list[index])
    print(hp_list)

    best_hp_model = os.path.join(dir_path, hp_list[index],"seed{}".format(seed), "model_{}.pt".format(args.epoch))
    model = torch.load(best_hp_model)
    evaluator = d3rlpy.metrics.EnvironmentEvaluator(env,gamma=0.995,n_trials=500)
    test_score_1_mean, test_score_1_std, test_score_mean, test_score_std, _ = evaluator(model)

    print("Epoch: {}".format(args.epoch))
    print("Method: {}".format(args.method))
    print("Seed: {}".format(args.seed))
    print("-"*30)
    
    print("Return mean when gamma = 1.0:", test_score_1_mean)
    print("Return std when gamma = 1.0:", test_score_1_std)
    print("Return mean when gamma = 0.995:", test_score_mean)
    print("Return std when gamma = 0.995:", test_score_std)
    
    

if __name__ == "__main__":
    main()