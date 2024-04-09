from d3rlpy.utils import save_policy, run
import torch
import os
import pdb

def process_baseline1(save_dir, dataset, python_file, eval_file, lr, algo):
    dir_path = "{}/baseline1/{}".format(save_dir, dataset)
    hp_list = os.listdir(dir_path)

    for hp in hp_list:
        for seed in range(1,6):
            for epoch in range(60,110,10):
                seed_dir_path = os.path.join(dir_path, hp, "seed{}".format(seed))

                # save model to pkl
                model = torch.load(os.path.join(seed_dir_path, "model_{}.pt".format(epoch)))
                save_policy(model, seed_dir_path, epoch)

                # estimate
                run(device=0,
                    python_file=python_file,
                    eval_file=eval_file,
                    save_dir=save_dir,
                    env_name=dataset,
                    lr=lr,
                    policy_path="{}/policy_{}.pkl".format(seed_dir_path, epoch),
                    seed=seed,
                    algo=algo,
                    epoch=epoch)


def process_baseline2(save_dir, dataset, python_file, eval_file, lr, algo):
    dir_path = "{}/baseline2/{}/{}".format(save_dir, dataset, algo)
    hp_list = os.listdir(dir_path)

    for hp in hp_list:
        for seed in range(1,6):
            for epoch in range(60,110,10):
                seed_dir_path = os.path.join(dir_path, hp, "seed{}".format(seed))

                # save model to pkl
                model = torch.load(os.path.join(seed_dir_path, "model_{}.pt".format(epoch)))
                save_policy(model, seed_dir_path, epoch)

                # estimate
                run(device=0,
                    python_file=python_file,
                    eval_file=eval_file,
                    save_dir=save_dir,
                    env_name=dataset,
                    lr=lr,
                    policy_path="{}/policy_{}.pkl".format(seed_dir_path, epoch),
                    seed=seed,
                    algo=algo,
                    epoch=epoch)