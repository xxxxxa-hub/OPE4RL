import subprocess

def run(device, python_file, eval_file, save_dir, env_name, lr, policy_path, seed, algo):
    commands = ["CUDA_VISIBLE_DEVICES={} {} {} --save_dir={}\
                --env_name={} --d4rl_policy_filename={} \
                --target_policy_std=0.0 --seed={} --algo={} \
                --noise_scale=0.0 --lr={}".format(device, python_file, eval_file, save_dir, env_name, policy_path, seed, algo, lr)]

    for command in commands:
        subprocess.run(command, shell=True)

# if __name__ == "__main__":
#     device = 1
#     env_name = "halfcheetah-random-v0"
#     lr = 0.01
#     lr_decay = 0.86
#     seed = 0
#     algo = "fqe"
#     run(device=device,env_name=env_name,lr=lr,lr_decay=lr_decay,seed=seed,algo=algo)