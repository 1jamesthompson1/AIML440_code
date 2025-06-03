import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import argparse
import gymnasium as gym
from functools import partial

from sunrise import sunrise
import core

def parse_args():
    parser = argparse.ArgumentParser(description="CLI for SUNRISE algorithm job submission.")
    parser.add_argument('--exp-dir', type=str, default='.', help="Directory to save experiment logs and models.")
    parser.add_argument('--exp-name', type=str, help="Name of the experiment for logging.")
    
    # Environment
    parser.add_argument('--env', type=str, required=True, help="Environment name (e.g., 'Ant-v5').")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility.")
    
    # Actor-Critic
    parser.add_argument('--ac-arch', type=int, nargs='+', default=[256, 256], help="Actor and Critic network architecture.")
    parser.add_argument('--num-ensemble', type=int, default=3, help="Number of actor-critic models in the ensemble.")
    
    # Training
    parser.add_argument('--total-steps', type=int, default=1000000, help="Total number of environment interactions.")
    parser.add_argument('--start-steps', type=int, default=10000, help="Steps of random action before policy starts.")
    parser.add_argument('--batch-size', type=int, default=256, help="Batch size for training.")
    parser.add_argument('--update-after', type=int, default=1000, help="Steps before starting gradient updates.")
    parser.add_argument('--update-every', type=int, default=50, help="Frequency of gradient updates.")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor.")
    parser.add_argument('--polyak', type=float, default=0.995, help="Polyak averaging factor.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--alpha', type=float, default=0.2, help="Entropy regularization coefficient.")
    parser.add_argument('--autotune-alpha', action='store_true', help="Enable automatic tuning of alpha.")
    parser.add_argument('--target-entropy', type=float, default=None, help="Target entropy for alpha tuning.")
    
    # Logging and Saving
    parser.add_argument('--log-every', type=int, default=10000, help="Frequency of logging.")
    parser.add_argument('--save-freq', type=int, default=10000, help="Frequency of saving the model.")
    parser.add_argument('--save-path', type=str, default=None, help="Path to save the trained model.")
    
    # Evaluation
    parser.add_argument('--num-test-episodes', type=int, default=10, help="Number of test episodes.")
    parser.add_argument('--max-ep-len', type=int, default=1000, help="Maximum length of an episode.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # Actor-Critic kwargs
    ac_kwargs = dict(
        hidden_sizes=args.ac_arch,
        activation=tf.keras.activations.relu,
    )

    env_fn = partial(gym.make, args.env, max_episode_steps=args.max_ep_len)
    
    # Run SUNRISE
    sunrise(
        exp_dir=args.exp_dir,
        exp_name=args.exp_name,
        env_fn=env_fn,
        actor_critic=core.MLPActorCriticFactory,
        ac_kwargs=ac_kwargs,
        ac_number=args.num_ensemble,
        seed=args.seed,
        steps_per_epoch=args.total_steps // 100,
        total_steps=args.total_steps,
        log_every=args.log_every,
        replay_size=int(1e6),
        gamma=args.gamma,
        polyak=args.polyak,
        lr=args.lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        start_steps=args.start_steps,
        update_after=args.update_after,
        update_every=args.update_every,
        num_test_episodes=args.num_test_episodes,
        save_freq=args.save_freq,
        autotune_alpha=args.autotune_alpha,
        target_entropy=args.target_entropy,
    )

if __name__ == "__main__":
    main()
