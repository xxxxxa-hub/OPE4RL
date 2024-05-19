from typing import (
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)
from collections import defaultdict
import d3rlpy
from d3rlpy.dataset import D4rlDataset
from d3rlpy.logging import LOG
from d3rlpy.utils import run, save_policy
import torch
from torch.utils.data import DataLoader
import numpy as np
import wandb
import csv
from tqdm import tqdm
import os
import pandas as pd
from scipy.stats import norm
import pdb
from torch.multiprocessing import Process, Queue, set_start_method, Manager
import gym
import sys
import random
import time


class SynchronizedExperiment():
    def __init__(self, lr_range, encoder, batch_size, conservative_weight, device) -> None:
       self.lr_range = lr_range
       self.encoder = encoder
       self.batch_size = batch_size
       self.conservative_weight = conservative_weight
       self.device = device
       self.runs_id = defaultdict()
       self.policies = defaultdict()
       self.random_states = defaultdict()
       self.np_states = defaultdict()
       self.torch_states = defaultdict()
       self.torch_cuda_states = defaultdict()
    
    def train_baseline_one_epoch(self, lr, cql, dataset, batch_size, dir_path, env_name, 
                                 save_dir, algo, seed, python_file, eval_file, estimator_lr, epoch, n_epoch, 
                                 collect_epoch, device, random_state, np_state, torch_state, torch_cuda_state, 
                                 results_queue):
        # dict to add incremental mean losses to epoch
        # os.environ["WANDB_MODE"] = "offline"
        _, env = d3rlpy.dataset.get_d4rl(env_name)
        # env = gym.make(env_name)
        d3rlpy.seed(seed)
        d3rlpy.envs.seed_env(env, seed)

        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state(torch_cuda_state)
        np.random.set_state(np_state)
        random.setstate(random_state)

        evaluators = {"environment": d3rlpy.metrics.EnvironmentEvaluator(env, gamma=0.995, max_steps=2000)}
        
        epoch_loss = defaultdict(list)

        if dataset["terminals"][-1] != True:
            dataset["timeouts"][-1] = True

        behavior_dataset = D4rlDataset(
            dataset,
            normalize_states=False,
            normalize_rewards=False,
            noise_scale=0.0,
            bootstrap=False)

        dataloader = DataLoader(behavior_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        dataloader = iter(dataloader)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        range_gen = tqdm(
            range(len(behavior_dataset) // batch_size),
            disable=False,
            desc=f"Epoch {int(epoch)}/{n_epoch}",
        )

        for itr in range_gen:
            states, actions, next_states, rewards, masks, _, _ = next(dataloader)

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            masks = masks.to(device)

            loss = cql.update(states, actions, next_states, rewards, masks)
            # record metrics
            for name, val in loss.items():
                epoch_loss[name].append(val)

            # update progress postfix with losses
            if itr % 10 == 0:
                mean_loss = {
                    k: np.mean(v) for k, v in epoch_loss.items()
                }
                range_gen.set_postfix(mean_loss)
        
        # OPE
        save_policy(cql, dir_path)
        run(device=device.split(":")[-1],
            python_file=python_file,
            eval_file=eval_file,
            save_dir=save_dir,
            env_name=env_name,
            lr=estimator_lr,
            policy_path="{}/policy.pkl".format(dir_path),
            seed=seed,
            algo=algo)

        estimate = pd.read_csv("{}/ope.csv".format(dir_path)).iloc[0,0]

        # Upload to wandb
        
        if evaluators:
            for name, evaluator in evaluators.items():
                test_score_1, _, test_score, _, transitions = evaluator(cql)

        with open("{}/loss.csv".format(dir_path), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([np.mean(epoch_loss["actor_loss"]),
                             np.mean(epoch_loss["critic_loss"]),
                             np.mean(epoch_loss["temp_loss"]),
                             np.mean(epoch_loss["temp"]),
                             estimate,
                             test_score_1,
                             test_score])
            
        # wandb.init(project="{}-{}-{}".format(env_name, method, save_dir.split("_")[-1]),
        #     name="Baseline-{}-{}-{}".format(lr, algo, seed),
        #     config={"learning_rate": lr, "algo": algo, "seed": seed},
        #     reinit=True,
        #     resume="allow",
        #     id=self.runs_id[lr])


        # if upload:
        #     wandb.log({"actor_loss": np.mean(epoch_loss["actor_loss"]),
        #         "critic_loss": np.mean(epoch_loss["critic_loss"]),
        #         "temp_loss": np.mean(epoch_loss["temp_loss"]),
        #         "temp": np.mean(epoch_loss["temp"]),
        #         "Oracle_1.0": test_score_1,
        #         "Oracle_0.995": test_score,
        #         "Estimate": estimate})

        if epoch % 10 == 0 and epoch > collect_epoch:
            torch.save(cql, "{}/model_{}.pt".format(dir_path, epoch))

    # Add estimate value and collected transitions
        torch_state = torch.get_rng_state().tolist()
        torch_cuda_state = torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        random_state = random.getstate()
        results_queue.put([lr, estimate, random_state, np_state, torch_state, torch_cuda_state])
        

    
    def train_incentive_one_epoch(self, lr, cql, dataset, batch_size, dir_path, env_name, test_score_collect_epoch,
                                 save_dir, algo, seed, python_file, eval_file, estimator_lr, epoch, n_epoch, reward_incentive,
                                 collect_epoch, device, random_state, np_state, torch_state, torch_cuda_state, reward_incentive_list,
                                 results_queue):
        # dict to add incremental mean losses to epoch
        # os.environ["WANDB_MODE"] = "offline"
        # env = gym.make(env_name)
        _, env = d3rlpy.dataset.get_d4rl(env_name)
        d3rlpy.seed(seed)
        d3rlpy.envs.seed_env(env, seed)

        torch.set_rng_state(torch_state)
        torch.cuda.set_rng_state(torch_cuda_state)
        np.random.set_state(np_state)
        random.setstate(random_state)

        evaluators = {"environment": d3rlpy.metrics.EnvironmentEvaluator(env, gamma=0.995, max_steps=2000)}
        
        epoch_loss = defaultdict(list)

        if dataset["terminals"][-1] != True:
            dataset["timeouts"][-1] = True

        behavior_dataset = D4rlDataset(
            dataset,
            normalize_states=False,
            normalize_rewards=False,
            noise_scale=0.0,
            bootstrap=False)

        dataloader = DataLoader(behavior_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        dataloader = iter(dataloader)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if len(reward_incentive_list) >= 10:
            reward_incentive_reduce_mean = reward_incentive - torch.tensor(np.mean(reward_incentive_list))
            normalized_reward_incentive = reward_incentive_reduce_mean / torch.std(torch.tensor(reward_incentive_list), unbiased=False)
        elif len(reward_incentive_list) < 10:
            normalized_reward_incentive = torch.tensor(0.0,dtype=torch.float32,device=self.device)

        cdf_value = norm.cdf(normalized_reward_incentive.cpu(), loc=0.0, scale=1.0)
        cdf_weight = 0.95 ** (epoch - collect_epoch) if epoch > collect_epoch else 1.0

        range_gen = tqdm(
            range(len(behavior_dataset) // batch_size),
            disable=False,
            desc=f"Epoch {int(epoch)}/{n_epoch}",
        )

        for itr in range_gen:
            states, actions, next_states, rewards, masks, _, _ = next(dataloader)

            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            masks = masks.to(device)

            loss = cql.update(states=states, actions=actions, next_states=next_states, 
                                    rewards=rewards.min() + 2 * (rewards - rewards.min()) * 
                                    (cdf_weight * cdf_value + (1 - cdf_weight) * 0.5), 
                                    masks=masks) # rewards + temp * normalized_reward_incentive

            # record metrics
            for name, val in loss.items():
                epoch_loss[name].append(val)

            # update progress postfix with losses
            if itr % 10 == 0:
                mean_loss = {
                    k: np.mean(v) for k, v in epoch_loss.items()
                }
                range_gen.set_postfix(mean_loss)
        
        # OPE
        save_policy(cql, dir_path)
        run(device=device.split(":")[-1],
            python_file=python_file,
            eval_file=eval_file,
            save_dir=save_dir,
            env_name=env_name,
            lr=estimator_lr,
            policy_path="{}/policy.pkl".format(dir_path),
            seed=seed,
            algo=algo)

        estimate = pd.read_csv("{}/ope.csv".format(dir_path)).iloc[0,0]

        # Upload to wandb
        if evaluators:
            for name, evaluator in evaluators.items():
                test_score_1, _, test_score, _, transitions = evaluator(cql)

            if epoch == collect_epoch:
                test_score_collect_epoch = test_score

        with open("{}/loss.csv".format(dir_path), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([np.mean(epoch_loss["actor_loss"]),
                             np.mean(epoch_loss["critic_loss"]),
                             np.mean(epoch_loss["temp_loss"]),
                             np.mean(epoch_loss["temp"]),
                             normalized_reward_incentive,
                             cdf_value,
                             estimate,
                             test_score_1,
                             test_score])
            
        # wandb.init(project="{}-{}-{}".format(env_name, method, save_dir.split("_")[-1]),
        #     name="Baseline-{}-{}-{}".format(lr, algo, seed),
        #     config={"learning_rate": lr, "algo": algo, "seed": seed},
        #     reinit=True,
        #     resume="allow",
        #     id=self.runs_id[lr])


        # if upload:
        #     wandb.log({"actor_loss": np.mean(epoch_loss["actor_loss"]),
        #         "critic_loss": np.mean(epoch_loss["critic_loss"]),
        #         "temp_loss": np.mean(epoch_loss["temp_loss"]),
        #         "temp": np.mean(epoch_loss["temp"]),
        #         "Oracle_1.0": test_score_1,
        #         "Oracle_0.995": test_score,
        #         "Estimate": estimate})

        if epoch % 10 == 0 and epoch > collect_epoch:
            torch.save(cql, "{}/model_{}.pt".format(dir_path, epoch))

    # Add estimate value and collected transitions
        torch_state = torch.get_rng_state().tolist()
        torch_cuda_state = torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None
        np_state = np.random.get_state()
        random_state = random.getstate()

        if epoch > collect_epoch:
            test_score = test_score_collect_epoch

        reward_incentive = np.float32(abs(estimate - test_score * (1-evaluators["environment"]._gamma)))
        reward_incentive = -reward_incentive
        # store new reward_incentive into reward_list
        if reward_incentive not in reward_incentive_list:
            reward_incentive_list.append(reward_incentive)

        results_queue.put([lr, estimate, random_state, np_state, torch_state, torch_cuda_state, reward_incentive, reward_incentive_list, test_score_collect_epoch])

        

        

    def fit(self,
        method,
        dataset,
        n_epoch,
        evaluators,
        python_file,
        eval_file,
        save_dir,
        seed,
        env_name,
        collect_epoch,
        estimator_lr,
        algo,
        upload,
        collect
    ) -> List[Tuple[int, Dict[str, float]]]:
        if method == "baseline1":
            self.fitter_1(
                method=method,
                dataset=dataset,
                n_epoch=n_epoch,
                evaluators=evaluators,
                python_file=python_file,
                eval_file=eval_file,
                save_dir=save_dir,
                seed=seed,
                env_name=env_name,
                collect_epoch=collect_epoch,
                estimator_lr=estimator_lr,
                algo=algo,
                upload=upload,
                collect=collect
                )
        elif method == "baseline2":
            self.fitter_2(
                method=method,
                dataset=dataset,
                n_epoch=n_epoch,
                evaluators=evaluators,
                python_file=python_file,
                eval_file=eval_file,
                save_dir=save_dir,
                seed=seed,
                env_name=env_name,
                collect_epoch=collect_epoch,
                estimator_lr=estimator_lr,
                algo=algo,
                upload=upload,
                collect=collect
                )
    
    def fitter_1(
        self,
        method,
        dataset,
        n_epoch,
        evaluators,
        python_file,
        eval_file,
        save_dir,
        seed,
        env_name,
        collect_epoch,
        estimator_lr,
        algo,
        upload,
        collect
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        # os.environ["WANDB_MODE"] = "offline"

        torch_state = torch.get_rng_state()
        torch_cuda_state = torch.cuda.get_rng_state()
        np_state = np.random.get_state()
        random_state = random.getstate()

        for lr in self.lr_range:
            # new_run = wandb.init(
            #     project="{}-{}-{}".format(env_name, method, save_dir.split("_")[-1]),
            #     name="Baseline-{}-{}-{}".format(lr, algo, seed),
            #     config={"learning_rate": lr,
            #             "algo": algo,
            #             "seed": seed},
            #     reinit=True,
            #     resume="allow"
            # )
            # print(new_run.id)
            # self.runs_id[lr] = new_run.id

            cql = d3rlpy.algos.CQLConfig(
                actor_learning_rate=lr,
                critic_learning_rate=lr,
                temp_learning_rate=1e-4,
                actor_encoder_factory=self.encoder,
                critic_encoder_factory=self.encoder,
                batch_size=self.batch_size,
                n_action_samples=10,
                alpha_learning_rate=0.0,
                conservative_weight=self.conservative_weight,
            ).create(device=self.device)

            if cql._impl is None:
                LOG.debug("Building models...")
                action_size = evaluators["environment"]._env.unwrapped.action_space.shape[0]
                observation_shape = evaluators["environment"]._env.unwrapped.observation_space.shape
                cql.create_impl(observation_shape, action_size)
                LOG.debug("Models have been built.")
            else:
                LOG.warning("Skip building models since they're already built.")

            self.policies[lr] = cql
            self.random_states[lr] = random_state
            self.np_states[lr] = np_state
            self.torch_states[lr] = torch_state
            self.torch_cuda_states[lr] = torch_cuda_state

            dir_path = "{}/{}/{}/{}/seed{}".format(save_dir, method, env_name, lr, seed)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open("{}/loss.csv".format(dir_path), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["actor_loss",
                                "critic_loss",
                                "temp_loss",
                                "temp",
                                "estimate",
                                "test_score_1",
                                "test_score"])

        # save state_dict of cql instead of passing them into the Process
        
        with Manager() as manager:
            for epoch in range(1, n_epoch + 1):
                results_queue = manager.Queue()
                # results_queue = Queue()
                processes = []   
            

                for lr in self.lr_range:
                    dir_path = "{}/{}/{}/{}/seed{}".format(save_dir, method, env_name, lr, seed)

                    p = Process(target=self.train_baseline_one_epoch, kwargs={
                        'lr': lr,
                        'cql': self.policies[lr],
                        'batch_size': self.batch_size,
                        'device': self.device,
                        'dataset': dataset,
                        'dir_path': dir_path,
                        # 'upload': upload,
                        'env_name': env_name,
                        # 'method': method,
                        'save_dir': save_dir,
                        'algo': algo,
                        'seed': seed,
                        'python_file': python_file,
                        'eval_file': eval_file,
                        'estimator_lr': estimator_lr,
                        'epoch': epoch,
                        'n_epoch': n_epoch,
                        'collect_epoch': collect_epoch,
                        'random_state': self.random_states[lr],
                        'np_state': self.np_states[lr],
                        'torch_state': self.torch_states[lr],
                        'torch_cuda_state': self.torch_cuda_states[lr],
                        'results_queue': results_queue
                    })
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
                print("Finish Joining")
                

                # Select the best policy based on estimate
                best_lr = None
                best_estimate = float('-inf')
                while not results_queue.empty():
                    lr, estimate, random_state, np_state, torch_state, torch_cuda_state = results_queue.get()

                    torch_state = torch.ByteTensor(torch_state)
                    torch_cuda_state = torch.ByteTensor(torch_cuda_state)
                    self.random_states[lr] = random_state
                    self.np_states[lr] = np_state
                    self.torch_states[lr] = torch_state
                    self.torch_cuda_states[lr] = torch_cuda_state

                    if estimate > best_estimate:
                        best_lr, best_estimate = lr, estimate

                # Add transitions to dataset
                print("Collecting Transitions")
                if evaluators:
                    for name, evaluator in evaluators.items():
                        test_score_1, _, test_score, _, transitions = evaluator(self.policies[best_lr])
                if collect and epoch <= collect_epoch:
                    for k,v in transitions.items():
                        dataset[k] = np.append(dataset[k], v, axis=0)
                        

    def fitter_2(
        self,
        method,
        dataset,
        n_epoch,
        evaluators,
        python_file,
        eval_file,
        save_dir,
        seed,
        env_name,
        collect_epoch,
        estimator_lr,
        algo,
        upload,
        collect
    ) -> Generator[Tuple[int, Dict[str, float]], None, None]:
        """Iterate over epochs steps to train with the given dataset. At each
        iteration algo methods and properties can be changed or queried.

        .. code-block:: python

            for epoch, metrics in algo.fitter(episodes):
                my_plot(metrics)
                algo.save_model(my_path)

        Args:
            dataset: Offline dataset to train.
            n_steps: Number of steps to train.
            n_steps_per_epoch: Number of steps per epoch. This value will
                be ignored when ``n_steps`` is ``None``.
            experiment_name: Experiment name for logging. If not passed,
                the directory name will be `{class name}_{timestamp}`.
            with_timestamp: Flag to add timestamp string to the last of
                directory name.
            logger_adapter: LoggerAdapterFactory object.
            show_progress: Flag to show progress bar for iterations.
            evaluators: List of evaluators.
            callback: Callable function that takes ``(algo, epoch, total_step)``
                , which is called every step.
            epoch_callback: Callable function that takes
                ``(algo, epoch, total_step)``, which is called at the end of
                every epoch.

        Returns:
            Iterator yielding current epoch and metrics dict.
        """
        reward_incentive_lrs = defaultdict(float)
        reward_incentive_list_lrs = defaultdict(list)
        test_score_collect_epoch_lrs = defaultdict(float)

        torch_state = torch.get_rng_state()
        torch_cuda_state = torch.cuda.get_rng_state()
        np_state = np.random.get_state()
        random_state = random.getstate()

        for lr in self.lr_range:
            # new_run = wandb.init(
            #     project="{}-{}-{}".format(env_name, method, save_dir.split("_")[-1]),
            #     name="Baseline-{}-{}-{}".format(lr, algo, seed),
            #     config={"learning_rate": lr,
            #             "algo": algo,
            #             "seed": seed},
            #     reinit=True,
            #     resume="allow"
            # )
            # print(new_run.id)
            # self.runs_id[lr] = new_run.id

            cql = d3rlpy.algos.CQLConfig(
                actor_learning_rate=lr,
                critic_learning_rate=lr,
                temp_learning_rate=1e-4,
                actor_encoder_factory=self.encoder,
                critic_encoder_factory=self.encoder,
                batch_size=self.batch_size,
                n_action_samples=10,
                alpha_learning_rate=0.0,
                conservative_weight=self.conservative_weight,
            ).create(device=self.device)

            if cql._impl is None:
                LOG.debug("Building models...")
                action_size = evaluators["environment"]._env.unwrapped.action_space.shape[0]
                observation_shape = evaluators["environment"]._env.unwrapped.observation_space.shape
                cql.create_impl(observation_shape, action_size)
                LOG.debug("Models have been built.")
            else:
                LOG.warning("Skip building models since they're already built.")
            
            self.policies[lr] = cql
            self.random_states[lr] = random_state
            self.np_states[lr] = np_state
            self.torch_states[lr] = torch_state
            self.torch_cuda_states[lr] = torch_cuda_state

            dir_path = "{}/{}/{}/{}/seed{}".format(save_dir, method, env_name, lr, seed)

            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            with open("{}/loss.csv".format(dir_path), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["actor_loss",
                                "critic_loss",
                                "temp_loss",
                                "temp",
                                "normalized_reward_incentive",
                                "cdf_value",
                                "estimate",
                                "test_score_1",
                                "test_score"])

        # save state_dict of cql instead of passing them into the Process
        
        with Manager() as manager:
            for epoch in range(1, n_epoch + 1):
                results_queue = manager.Queue()
                processes = []   
            
                for lr in self.lr_range:
                    dir_path = "{}/{}/{}/{}/seed{}".format(save_dir, method, env_name, lr, seed)

                    p = Process(target=self.train_incentive_one_epoch, kwargs={
                        'lr': lr,
                        'cql': self.policies[lr],
                        'batch_size': self.batch_size,
                        'device': self.device,
                        'dataset': dataset,
                        'dir_path': dir_path,
                        # 'upload': upload,
                        'reward_incentive': reward_incentive_lrs[lr],
                        'reward_incentive_list': reward_incentive_list_lrs[lr],
                        'test_score_collect_epoch': test_score_collect_epoch_lrs[lr],
                        'env_name': env_name,
                        # 'method': method,
                        'save_dir': save_dir,
                        'algo': algo,
                        'seed': seed,
                        'python_file': python_file,
                        'eval_file': eval_file,
                        'estimator_lr': estimator_lr,
                        'epoch': epoch,
                        'n_epoch': n_epoch,
                        'collect_epoch': collect_epoch,
                        'random_state': self.random_states[lr],
                        'np_state': self.np_states[lr],
                        'torch_state': self.torch_states[lr],
                        'torch_cuda_state': self.torch_cuda_states[lr],
                        'results_queue': results_queue
                    })
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
                print("Finish Joining")
                

                # Select the best policy based on estimate
                best_lr = None
                best_estimate = float('-inf')
                while not results_queue.empty():
                    lr, estimate, random_state, np_state, torch_state, torch_cuda_state, reward_incentive, \
                    reward_incentive_list, test_score_collect_epoch = results_queue.get()

                    torch_state = torch.ByteTensor(torch_state)
                    torch_cuda_state = torch.ByteTensor(torch_cuda_state)
                    self.random_states[lr] = random_state
                    self.np_states[lr] = np_state
                    self.torch_states[lr] = torch_state
                    self.torch_cuda_states[lr] = torch_cuda_state
                    reward_incentive_lrs[lr] = reward_incentive
                    reward_incentive_list_lrs[lr] = reward_incentive_list
                    test_score_collect_epoch_lrs[lr] = test_score_collect_epoch

                    if estimate > best_estimate:
                        best_lr, best_estimate = lr, estimate

                # Add transitions to dataset
                print("Collecting Transitions")
                if evaluators:
                    for name, evaluator in evaluators.items():
                        test_score_1, _, test_score, _, transitions = evaluator(self.policies[best_lr])
                if collect and epoch <= collect_epoch:
                    for k,v in transitions.items():
                        dataset[k] = np.append(dataset[k], v, axis=0)