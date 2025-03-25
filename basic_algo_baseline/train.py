
from functools import partial
from itertools import product
import gymnasium as gym
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import argparse
import os

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument('--seed', required=True, type=int, help='Random seed for reproducibility')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the model')
parser.add_argument('--algo', required=True, type=str, choices=['sac', 'ppo', 'td3'] ,help='Algorithm to use for training')
parser.add_argument('--env_index', required=True, type=int, help='Environment to train on')

args = parser.parse_args()

os.chdir(args.output_dir)


def learn_and_save(model, name, eval_env, **kwargs):
    print(f"Training {name} inside {os.getcwd()}...")
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(eval_env, eval_freq=1000, n_eval_episodes=10, log_path=None, best_model_save_path=None)

    # Manually set the logpath because I want to name the output file
    eval_callback.log_path = f"training_evaluations/{name}.npz"
    model.learn(callback=eval_callback, **kwargs)
    # Save the evaluation results
    model.save("models/" + name)


time_steps = 1_000_000

tensorboard_logs = "tensorboard_logs"

# Using the hyperparameters from the original papers to get as close as possible
algorithms = {
    "sac": partial(PPO, "MlpPolicy", learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, verbose=0, tensorboard_log = tensorboard_logs),
    "ppo": partial(SAC, "MlpPolicy", learning_rate=0.0003, buffer_size=1e6, learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, verbose=0, tensorboard_log = tensorboard_logs),
    "td3": partial(TD3, "MlpPolicy", learning_rate=0.001, buffer_size=1000000, learning_starts=1000, batch_size=100, tau=0.005, gamma=0.99, train_freq=1, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, gradient_steps=1, verbose=0, tensorboard_log = tensorboard_logs),
}

envs = [
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "Ant-v5",
    "HumanoidStandup-v5",
    "Swimmer-v5",
    "Hopper-v5",
    "InvertedDoublePendulum-v5",
    "Pusher-v5",
]

os.makedirs("models", exist_ok=True)
os.makedirs("logs/avg_reward", exist_ok=True)

env = envs[args.env_index]

learn_and_save(
    algorithms[args.algo](gym.make(env), seed=args.seed),
    f"{args.algo}_{env}_{time_steps}_{args.seed}",
    gym.make(env),
    total_timesteps=time_steps,
)