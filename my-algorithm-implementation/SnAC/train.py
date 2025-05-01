import numpy as np
import torch
import argparse
from typing import List, Tuple, Optional, Any
from pathlib import Path
import time
import random
import gymnasium as gym
import os.path as osp
import re
import subprocess


from snac import SnAC


def train_snac(
    env_name: str,
    num_actors: int,
    gamma: float,
    tau: float,
    lr_q: float,
    lr_pi: float,
    lr_alpha: float,
    alpha_div: float,
    batch_size: int,
    memory_size: int,
    hidden_dim: int,
    auto_entropy: bool,
    updates_per_step: int,
    seed: int,
    time_steps: int,
    start_steps: int,
    eval_every: int,
    eval_dir: str,
    save_model: bool,
    save_model_path: str,
    experiment_name: str,
):
    """
    Trains the Soft n Actor-Critic (SnAC) algorithm on a specified environment.
    """
    # --- Seeding ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
        # Potentially add deterministic flags, though they can impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env: gym.Env = gym.make(env_name)

    agent = SnAC(
        env,
        num_actors=num_actors,
        gamma=gamma,
        tau=tau,
        lr_q=lr_q,
        lr_pi=lr_pi,
        lr_alpha=lr_alpha,
        alpha_div=alpha_div,
        batch_size=batch_size,
        memory_size=memory_size,
        hidden_dim=hidden_dim,
        auto_entropy=auto_entropy,
        target_update_interval=1,
        computation_device=DEVICE,
    )

    state_tuple: Tuple[np.ndarray, dict] = env.reset(seed=seed)
    state: np.ndarray = state_tuple[0]
    episode_reward: float = 0.0
    episode_timesteps: int = 0
    episode_num: int = 0
    episode_start_time: float = time.time()

    # Setting up evaluation variables
    eval_rewards: List[List[float]] = []
    eval_ep_lengths: List[List[int]] = []
    eval_time_steps: List[int] = []
    eval_path = Path(eval_dir)
    eval_path.mkdir(parents=True, exist_ok=True)

    
    print ("Training parameters:")
    print(f"Environment: {env_name}")
    print(f"Device: {DEVICE}")
    print(f"Environment name: {env_name}")
    print(f"Parameters: num_actors={num_actors}, gamma={gamma}, tau={tau}, lr_q={lr_q}, lr_pi={lr_pi}, lr_alpha={lr_alpha}, alpha_div={alpha_div}, batch_size={batch_size}, memory_size={memory_size}, hidden_dim={hidden_dim}, auto_entropy={auto_entropy}, updates_per_step={updates_per_step}, seed={seed}, time_steps={time_steps}, start_steps={start_steps}, eval_every={eval_every}, eval_dir={eval_dir}, save_model={save_model}, save_model_path={save_model_path}, experiment_name={experiment_name}")
    print(f"Start time of training: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(episode_start_time))}")


    for frame_idx in range(1, time_steps + 1):
        episode_timesteps += 1

        action: np.ndarray
        policy_index: int
        if frame_idx < start_steps:
            action = env.action_space.sample()
            policy_index = -1  # Indicate random action
        else:
            action, policy_index = agent.select_action(state)

        next_state: np.ndarray
        reward: Any
        terminated: bool
        truncated: bool
        next_state, reward, terminated, truncated, _ = env.step(action)
        done: bool = terminated or truncated

        # Ensure reward is float
        reward_float = float(reward)

        agent.memory.push(state, action, reward_float, next_state, done, policy_index)

        state = next_state
        episode_reward += reward_float

        if frame_idx >= start_steps:
            agent.update_parameters(updates_per_step)

        if done:
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            print(
                f"Total T: {frame_idx} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Alpha: {agent.alpha:.4f} Duration: {episode_duration:.2f}s"
            )
            state_tuple = env.reset(seed=seed + episode_num + 1)
            state = state_tuple[0]
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
            episode_start_time = time.time()

        if frame_idx % eval_every == 0 and frame_idx > start_steps:
            print(f"\n--- Evaluation at step {frame_idx} ---")
            evaluation_start_time = time.time()

            curr_eval_rewards, curr_eval_ep_lengths = evaluate_model(
                env_name=env_name,
                agent=agent,
                num_episodes=10,
                max_steps=1000,
                seed=seed + frame_idx,
            )

            if curr_eval_rewards:
                avg_reward: float = float(np.mean(curr_eval_rewards))
                print(
                    f"Average Evaluation Reward ({len(curr_eval_rewards)} episodes): {avg_reward:.3f}, evaluation duration: {time.time() - evaluation_start_time:.2f}s"
                )
            else:
                print("Evaluation failed or produced no results.")

            eval_rewards.append(curr_eval_rewards)
            eval_time_steps.append(frame_idx)
            eval_ep_lengths.append(curr_eval_ep_lengths)
            np.savez(
                eval_path / experiment_name,
                timesteps=eval_time_steps,
                results=eval_rewards,
                ep_lengths=eval_ep_lengths,
            )

            if save_model: # Use save_model parameter
                agent.save_model(osp.join(save_model_path, f"{experiment_name}.pt"))

    env.close()
    print("Training finished.")
    agent.save_model(osp.join(save_model_path, f"{experiment_name}_final.pt"))


def evaluate_model(env_name: str, agent: SnAC, seed: int, num_episodes: int=10, max_steps: int=1000):
    """
    Evaluates the model on a envrionment

    Args:
        env_name (str): Name of the environment to evaluate on. Made using gym.make(env_name)
        agent (SnAC): The SnAC agent to evaluate.
        seed (int): Random seed for reproducibility.
        num_episodes (int): Number of episodes to evaluate.
        max_steps (int): Maximum number of steps per episode, it will use the environments maximum number of steps if it has one.
    """
    curr_eval_rewards: List[float] = []
    curr_eval_ep_lengths: List[int] = []

    eval_env: Optional[gym.Env] = None
    try:
        eval_env = gym.make(env_name)
        for eval_ep_idx in range(num_episodes):
            eval_state_tuple = eval_env.reset(
                seed=seed + eval_ep_idx
            )
            eval_state = eval_state_tuple[0]
            eval_episode_reward: float = 0.0
            eval_terminated: bool = False
            eval_truncated: bool = False
            eval_steps: int = 0
            max_eval_steps = getattr(eval_env, "_max_episode_steps", max_steps)

            while (
                not (eval_terminated or eval_truncated)
                and eval_steps < max_eval_steps
            ):
                eval_action, _ = agent.select_action(eval_state, evaluate=True)
                eval_step_result = eval_env.step(eval_action)
                eval_state, eval_reward, eval_terminated, eval_truncated, _ = (
                    eval_step_result
                )
                eval_episode_reward += float(eval_reward)
                eval_steps += 1
            curr_eval_rewards.append(eval_episode_reward)
            curr_eval_ep_lengths.append(eval_steps)
    finally:
        if eval_env is not None:
            eval_env.close()

    return curr_eval_rewards, curr_eval_ep_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Soft n Actor-Critic (SnAC)

This is a PyTorch implementation of the Soft n Actor-Critic (SnAC) algorithm for continuous action spaces.

The CLI training implmentation is designed to be run in the directory where you want its output to be saved.
See the experiments direcotry of this project to see how I ran experiments using a SGE cluster and slurm.
""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

    )
    parser.add_argument(
        "--env-name",
        default="InvertedPendulum-v5",
        help="Gym environment name (default: InvertedPendulum-v5)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Base directory for the experiment data."
    )
    parser.add_argument(
        "--num-actors",
        type=int,
        default=3,
        metavar="N",
        help="Number of actors (default: 3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="Discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        metavar="T",
        help="Target network update rate (default: 0.005)",
    )
    parser.add_argument(
        "--lr-q",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Learning rate for Q-networks (default: 3e-4)",
    )
    parser.add_argument(
        "--lr-pi",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Learning rate for policy networks (default: 3e-4)",
    )
    parser.add_argument(
        "--lr-alpha",
        type=float,
        default=3e-4,
        metavar="LR",
        help="Learning rate for alpha (entropy tuning) (default: 3e-4)",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=100000,
        metavar="N",
        help="Replay buffer size (default: 100000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="Batch size for training (default: 256)",
    )
    parser.add_argument(
        "--updates-per-step",
        type=int,
        default=1,
        metavar="N",
        help="Model updates per environment step (default: 1)",
    )
    parser.add_argument(
        "--start-steps",
        type=int,
        default=10000,
        metavar="N",
        help="Number of steps with random actions at the beginning (default: 10000)",
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Maximum number of environment steps (default: 1000000)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1_000, # Corrected default value based on help text
        metavar="N",
        help="Evaluate policy every N steps (default: 10000)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        metavar="N",
        help="Hidden layer dimension (default: 256)",
    )
    parser.add_argument(
        "--alpha-div",
        type=float,
        default=0.01,
        metavar="A",
        help="Diversity coefficient strength (default: 0.01)",
    )
    parser.add_argument(
        "--auto-entropy",
        action="store_true",
        default=True,
        help="Automatically tune entropy (alpha)",
    )
    parser.add_argument(
        "--no-auto-entropy",
        action="store_false",
        dest="auto_entropy",
        help="Do not automatically tune entropy (alpha)",
    )
    parser.add_argument(
        "--seed", type=int, default=123, metavar="N", help="Random seed (default: 123)"
    )
    parser.add_argument(
        "--eval-dir",
        type=str,
        default="training_evaluations",
        help="Directory for evaluation logs to go to (default: training_evaluations)",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Whether the model should be saved (default: True). Note depending on the environment this may be large as the replay buffer is saved as well.",
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="snac_model",
        help="Path prefix to save models (default: snac_model)",
    )
    parser.add_argument(
        "--computation-device",
        type=str,
        default="auto",
        help="What device to run the tranining process. Auto will use cuda if available otherwise use cpu, it will search for a cuda device that is not in use."
    )


    args = parser.parse_args()

    experiment_name = f"snac_{args.env_name}_{args.time_steps}_{args.seed}"

    # --- Configuration ---
    DEVICE: torch.device
    if args.computation_device == "auto":
        print("Automatically figuring out which GPU to use.")
        if torch.cuda.is_available():
            try:
                # Try to find the least utilized GPU
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
                )
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    index, memory_used = map(int, line.split(', '))
                    gpus.append({'index': index, 'memory_used': memory_used})

                print(f"GPUs available are:\n{gpus}")

                if gpus:
                    gpus.sort(key=lambda x: x['memory_used'])
                    best_gpu = gpus[0]
                    # Ignoring GPUs that have more than 10mb of memory used based on the heuristic that that would make them 'occupied'
                    if best_gpu['memory'] > 10:
                       print("All GPUs are occupied, using cpu") 
                       DEVICE = torch.device("cpu")
                    else:
                        DEVICE = torch.device(f"cuda:{best_gpu['index']}")
                        print(f"Auto-selected CUDA device: cuda:{best_gpu['index']}")
                else:
                    print("nvidia-smi found no GPUs, defaulting to cuda:0")
                    DEVICE = torch.device("cuda:0")

            except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
                print(f"Could not run nvidia-smi or parse output ({e}), defaulting to cuda:0")
                DEVICE = torch.device("cuda:0")
        else:
            print("CUDA not available, using CPU.")
            DEVICE = torch.device("cpu")
    else:
        try:
            DEVICE = torch.device(args.computation_device)
            print(f"Using specified device: {DEVICE}")
        except RuntimeError as e:
            print(f"Error setting device to '{args.computation_device}': {e}.")
            exit(1)


    eval_dir = osp.join(args.data_dir, args.eval_dir)
    save_model_path = osp.join(args.data_dir, args.save_model_path)
    Path(eval_dir).mkdir(parents=True, exist_ok=True)
    Path(save_model_path).mkdir(parents=True, exist_ok=True)

    train_snac(
        env_name=args.env_name,
        num_actors=args.num_actors,
        gamma=args.gamma,
        tau=args.tau,
        lr_q=args.lr_q,
        lr_pi=args.lr_pi,
        lr_alpha=args.lr_alpha,
        alpha_div=args.alpha_div,
        batch_size=args.batch_size,
        memory_size=args.memory_size,
        hidden_dim=args.hidden_dim,
        auto_entropy=args.auto_entropy,
        updates_per_step=args.updates_per_step,
        seed=args.seed,
        time_steps=args.time_steps,
        start_steps=args.start_steps,
        eval_every=args.eval_every,
        eval_dir=eval_dir,
        save_model=args.save_model,
        save_model_path=save_model_path,
        experiment_name=experiment_name,
    )
