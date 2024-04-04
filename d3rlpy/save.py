from d3rlpy.utils import save_policy, run
import torch
import os
import pdb

def process_baseline1(save_dir, dataset, python_file, eval_file, lr, lr_decay, algo):
    dir_path = "{}/baseline1/{}".format(save_dir, dataset)
    hp_list = os.listdir(dir_path)

    for hp in hp_list:
        for seed in range(1,6):
            seed_dir_path = os.path.join(dir_path, hp, "seed{}".format(seed))

            # save model to pkl
            model = torch.load(os.path.join(seed_dir_path, "model_100.pt"))
            save_policy(model, seed_dir_path)

            # estimate
            run(device=0,
                python_file=python_file,
                eval_file=eval_file,
                env_name=dataset,
                lr=lr,
                policy_path="{}/policy.pkl".format(seed_dir_path),
                lr_decay=lr_decay,
                seed=seed,
                algo=algo)

            # on-policy
            # rollout 100 episodes and save in oracle.csv
