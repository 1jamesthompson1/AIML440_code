from functools import partial
import gymnasium as gym
from stable_baselines3 import PPO, TD3, SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sbx import CrossQ, SAC as SBX_SAC
import argparse
import os
from datetime import datetime



def learn_and_save(model, name, eval_env, eval_output, model_output, **kwargs):
    print(f"Training {name} inside {os.getcwd()}...")
    eval_env = Monitor(eval_env)
    eval_callback = EvalCallback(eval_env, eval_freq=1000, n_eval_episodes=10, log_path=None, best_model_save_path=None)

    # Manually set the logpath because I want to name the output file
    eval_callback.log_path = os.path.join(eval_output, f"{name}.npz")
    model.learn(callback=eval_callback, **kwargs)
    # Save the evaluation results
    model.save(os.path.join(model_output, name))



def main():
    # Using the hyperparameters from the original papers to get as close as possible
    algorithms = {
        "ppo": partial(PPO, "MlpPolicy", learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95),
        "sac": partial(SAC, "MlpPolicy", learning_rate=0.0003, buffer_size=1000000, learning_starts=100, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1),
        "td3": partial(TD3, "MlpPolicy", learning_rate=0.001, buffer_size=1000000, learning_starts=1000, batch_size=100, tau=0.005, gamma=0.99, train_freq=1, policy_delay=2, target_policy_noise=0.2, target_noise_clip=0.5, gradient_steps=1),
        "crossq": partial(CrossQ, "MlpPolicy"),
        "droq": partial(SBX_SAC, "MlpPolicy", learning_rate=0.003, buffer_size=1000000, learning_starts=0, gradient_steps=20, policy_delay=20, policy_kwargs={"dropout_rate": 0.01, "layer_norm":True}),
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

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--seed', required=True, type=int, help='Random seed for reproducibility')
    parser.add_argument('--algo', required=True, type=str, choices=algorithms.keys() ,help='Algorithm to use for training')
    parser.add_argument('--env_index', required=True, type=int, help='Environment to train on')
    parser.add_argument('--time_steps', default=1e6, type=int, help='Number of time steps to train the model')

    ## Output
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the model')
    parser.add_argument('--model_output', type=str, default="models", help='Output directory for the model')
    parser.add_argument('--eval_output', type=str, default="training_evaluations", help='Output directory for the evaluation results')
    parser.add_argument('--tensorboard_log', type=str, default="tensorboard_logs", help='Output directory for the tensorboard logs')
    parser.add_argument('--verbosity', type=int, default=0, help='Verbosity level for the training process')

    args = parser.parse_args()

    os.chdir(args.output_dir)

    os.makedirs("models", exist_ok=True)

    env = envs[args.env_index]
    starttime = datetime.now()
    print(f"Start time of training: {starttime}")

    learn_and_save(
        algorithms[args.algo](gym.make(env), seed=args.seed, tensorboard_log=args.tensorboard_log, verbose=args.verbosity),
        f"{args.algo}_{env}_{args.time_steps}_{args.seed}",
        gym.make(env),
        total_timesteps=args.time_steps,
        eval_output=args.eval_output,
        model_output=args.model_output,
    )
    finishtime = datetime.now()
    print(f"End time of training: {finishtime}")
    print(f"Time taken for training: {finishtime - starttime}")
    print("Training finished.")

if __name__ == "__main__":
    main()