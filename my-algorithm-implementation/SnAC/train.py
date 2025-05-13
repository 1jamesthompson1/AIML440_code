import numpy as np
import torch
import argparse
from typing import List, Tuple, Optional, Any
from pathlib import Path
import time
import random
import gymnasium as gym
import os.path as osp
import os
import subprocess


from snac import SnAC


def train_snac(
    experiment_name: str,
    exp_dir: str,
    env_name: str,
    agent_params: dict,
    updates_per_step: int,
    seed: int,
    time_steps: int,
    start_steps: int,
    eval_every: int,
    save_model: bool,
    record_video: bool,
    computation_device: torch.device,
):
    """
    Trains the Soft n Actor-Critic (SnAC) algorithm on a specified environment.
    """
    # --- Seeding ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if computation_device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
        # Potentially add deterministic flags, though they can impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env: gym.Env = gym.make(env_name)

    agent = SnAC(
        env,
        **agent_params,
        target_update_interval=1,
        computation_device=computation_device,
    )

    if record_video:
        video_dir = osp.join(exp_dir, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)

    state_tuple: Tuple[np.ndarray, dict] = env.reset(seed=seed)
    state: np.ndarray = state_tuple[0]
    episode_reward: float = 0.0
    episode_timesteps: int = 0
    episode_num: int = 0
    episode_start_time: float = time.time()
    training_start_time: float = episode_start_time

    # Setting up evaluation variables
    eval_rewards: List[List[float]] = []
    eval_ep_lengths: List[List[int]] = []
    eval_time_steps: List[int] = []

    
    print ("Training parameters:")
    print(f"Environment: {env_name}")
    print(f"Device: {computation_device}")
    print(f"Process ID: {os.getpid()}")
    print(f"Environment name: {env_name}")
    print(f"Start time of training: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
    print(f"Using settings: updates_per_step={updates_per_step}, seed={seed}, time_steps={time_steps}, start_steps={start_steps}, eval_every={eval_every}, exp_dir={exp_dir}, save_model={save_model}, experiment_name={experiment_name}", flush=True)


    episode_action_counts: np.ndarray = np.zeros(agent.num_actors+1, dtype=int)
    for frame_idx in range(1, time_steps + 1):
        episode_timesteps += 1

        action: np.ndarray
        policy_index: int
        if frame_idx < start_steps:
            action = env.action_space.sample()
            policy_index = -1  # Indicate random action
        else:
            action, policy_index = agent.select_action(state)

        episode_action_counts[policy_index] += 1

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
                f"Total T: {frame_idx} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Alpha: {agent.alpha:.4f} Actor action counts: {episode_action_counts.tolist()} Duration: {episode_duration:.2f}s",
                 flush=True
            )
            state_tuple = env.reset(seed=seed + episode_num + 1)
            state = state_tuple[0]
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
            episode_start_time = time.time()
            episode_action_counts = np.zeros(agent.num_actors+1, dtype=int)

        if frame_idx % eval_every == 0 and frame_idx >= start_steps:
            print(f"\n--- Evaluation at step {frame_idx} ---")
            evaluation_start_time = time.time()

            curr_eval_rewards, curr_eval_ep_lengths = evaluate_model(
                env_name=env_name,
                agent=agent,
                num_episodes=10,
                max_steps=1000,
                seed=seed + frame_idx,
                video_dir=video_dir if record_video else None,
            )

            if curr_eval_rewards:
                avg_reward: float = float(np.mean(curr_eval_rewards))
                min_reward: float = float(np.min(curr_eval_rewards))
                max_reward: float = float(np.max(curr_eval_rewards))
                print(
                    f"Average Evaluation Reward ({len(curr_eval_rewards)} episodes): {avg_reward:.3f}, min: {min_reward:.3f}, max: {max_reward:.3f}, evaluation duration: {time.time() - evaluation_start_time:.2f}s", flush=True
                )
            else:
                print("Evaluation failed or produced no results.", flush=True)

            eval_rewards.append(curr_eval_rewards)
            eval_time_steps.append(frame_idx)
            eval_ep_lengths.append(curr_eval_ep_lengths)
            np.savez(
                osp.join(exp_dir, "evaluations"),
                timesteps=eval_time_steps,
                results=eval_rewards,
                ep_lengths=eval_ep_lengths,
            )

            if save_model: # Use save_model parameter
                agent.save_model(osp.join(exp_dir, "agent.pt"))

    env.close()
    end_training_time = time.time()
    total_training_duration = end_training_time - training_start_time

    print(
        f"Training completed in {total_training_duration:.2f}s. Which is {total_training_duration/3600} hours.",
    )
    agent.save_model(osp.join(exp_dir, "agent_final.pt"))


def evaluate_model(env_name: str, agent: SnAC, seed: int, num_episodes: int=10, max_steps: int=1000, video_dir = None):
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
        if video_dir is not None:
            eval_env = gym.make(env_name, render_mode="rgb_array")
            eval_env = gym.wrappers.RecordVideo(
                eval_env,
                video_folder=video_dir,
                episode_trigger=lambda ep_id: ep_id == 0,  # Only record the first episode
                name_prefix=f"eval-{seed}"
            )
        else:
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

def get_available_computation_device(min_free_memory_mib=1024, max_utilization_pct=99, power_margin_pct=10):
    """
    Checks for computation devices and selects the most suitable CUDA device based on utilization,
    memory availability, and power usage.

    Args:
        min_free_memory_mib (int): Minimum free memory required in MiB.
        max_utilization_pct (int): Maximum GPU utilization percentage allowed.
        power_margin_pct (int): Required power headroom percentage (e.g., 10 means power draw must be < 90% of limit).

    Returns:
        torch.device: The selected computation device (CPU or a specific CUDA device).
    """
    print("Automatically figuring out which GPU to use.")
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU.")
        return torch.device("cpu")

    try:
        # Query relevant GPU properties
        query_fields = "index,memory.used,memory.total,utilization.gpu,power.draw,power.limit"
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={query_fields}', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )

        gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line: continue
            try:
                # Parse values, handling potential missing data or formatting issues
                parts = line.split(', ')
                if len(parts) != 6:
                    print(f"Warning: Skipping malformed nvidia-smi line: {line}")
                    continue

                index = int(parts[0])
                memory_used = int(parts[1])
                memory_total = int(parts[2])
                utilization_gpu = int(parts[3])
                power_draw = float(parts[4])
                power_limit = float(parts[5])

                gpus.append({
                    'index': index,
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'memory_free': memory_total - memory_used,
                    'utilization_gpu': utilization_gpu,
                    'power_draw': power_draw,
                    'power_limit': power_limit,
                })
            except ValueError as e:
                print(f"Warning: Skipping line due to parsing error ({e}): {line}")
                continue


        print(f"GPUs found:\n{gpus}")

        available_gpus = []
        for gpu in gpus:
            # Check criteria
            is_memory_ok = gpu['memory_free'] >= min_free_memory_mib
            is_util_ok = gpu['utilization_gpu'] < max_utilization_pct
            # Check power margin (power_draw < (1 - power_margin_pct/100) * power_limit)
            is_power_ok = gpu['power_draw'] < (1.0 - power_margin_pct / 100.0) * gpu['power_limit']

            if is_memory_ok and is_util_ok and is_power_ok:
                available_gpus.append(gpu)
            else:
                print(f"GPU {gpu['index']} excluded: FreeMem={gpu['memory_free']} MiB (Req>{min_free_memory_mib}), Util={gpu['utilization_gpu']}% (Req<{max_utilization_pct}), PowerDraw={gpu['power_draw']:.1f}W (Limit={gpu['power_limit']:.1f}W, Margin={power_margin_pct}%)")


        if available_gpus:
            # Sort available GPUs by free memory (descending)
            available_gpus.sort(key=lambda x: x['memory_free'], reverse=True)
            best_gpu = available_gpus[0]
            device = torch.device(f"cuda:{best_gpu['index']}")
            print(f"Selected available CUDA device: cuda:{best_gpu['index']} (Free Memory: {best_gpu['memory_free']} MiB)")
            return device
        else:
            print("No suitable GPUs found based on criteria, using CPU.")
            return torch.device("cpu")

    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        print(f"Could not run nvidia-smi or parse output ({e}), defaulting to first available CUDA device or CPU")
        # Fallback logic: try generic cuda device if possible, else CPU
        if torch.cuda.is_available():
             try:
                 # Try allocating to default cuda device as a last resort
                 torch.zeros(1).to('cuda')
                 print("Warning: Falling back to default CUDA device.")
                 return torch.device("cuda")
             except Exception as cuda_err:
                 print(f"Warning: Default CUDA device also unavailable ({cuda_err}). Falling back to CPU.")
                 return torch.device("cpu")
        else:
             return torch.device("cpu")


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
        "--exp-name",
        help="This is the name of the experiment. Each of the outfiles will be called this."
    )
    parser.add_argument(
        "--env-name",
        default="InvertedPendulum-v5",
        help="Gym environment name (default: InvertedPendulum-v5)",
    )
    parser.add_argument(
        "--exp-dir",
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
        default=10_000,
        metavar="N",
        help="Evaluate policy every N steps (default: 10000)",
    )
    parser.add_argument(
        "--sticky-actor-actions",
        type=int,
        default=0,
        help="This decides how many action the select actor will get to take. This gives an actor some ability to 'carry out' its plan."
    )
    parser.add_argument(
        "--policy-update-temperature",
        type=float,
        default=0.1,
        help="This is the temperature for the policy update. It is used to determine how muc the policy disagrement affects the weight of the policy update."
    )
    parser.add_argument(
        '--actor-aware-critic',
        action='store_true',
        default=False,
        help='Use actor-aware critic (default: False). This means that the critic will be given the state, action taken AND the actor which took it (via one hot encoding)'
    )
    parser.add_argument(
        "--critic-arch",
        type=int,
        nargs='+',
        default=[256, 256],
        help="Architecture of the critic networks (list of hidden layer sizes)"
    )
    parser.add_argument(
        "--actor-arch",
        type=int,
        nargs='+',
        default=[256, 256],
        help="Architecture of the actor networks (list of hidden layer sizes)"
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
        "--save-model",
        action="store_true",
        default=True,
        help="Whether the model should be saved (default: True). Note depending on the environment this may be large as the replay buffer is saved as well.",
    )
    parser.add_argument(
        "--computation-device",
        type=str,
        default="auto",
        help="What device to run the tranining process. Auto will use cuda if available otherwise use cpu, it will search for a cuda device that is not in use."
    )
    parser.add_argument(
        "--aggregation-method",
        type=str,
        default="max_q",
        help="What aggregation method to use."
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        default=False,
        help="Directory to save evaluation videos. If None, no videos will be saved."
    )


    args = parser.parse_args()


    # --- Configuration ---
    computation_device: torch.device
    if args.computation_device == "auto":
        # Pass thresholds to the selection function if needed, or use defaults
        computation_device = get_available_computation_device(min_free_memory_mib=1024, max_utilization_pct=10, power_margin_pct=10)
    else:
        try:
            computation_device = torch.device(args.computation_device)
            print(f"Using specified device: {computation_device}")
        except RuntimeError as e:
            print(f"Error setting device to '{args.computation_device}': {e}.")
            exit(1)

    # --- Experiment output directory structure ---

    agent_params = {
        "num_actors": args.num_actors,
        "aggregation_method": args.aggregation_method,
        "gamma": args.gamma,
        "tau": args.tau,
        "lr_q": args.lr_q,
        "lr_pi": args.lr_pi,
        "lr_alpha": args.lr_alpha,
        "alpha_div": args.alpha_div,
        "batch_size": args.batch_size,
        "memory_size": args.memory_size,
        "critic_arch": args.critic_arch,
        "actor_arch": args.actor_arch,
        "auto_entropy": args.auto_entropy,
        "sticky_actor_actions": args.sticky_actor_actions,
        "policy_update_temperature": args.policy_update_temperature,
        "actor_aware_critic": args.actor_aware_critic,
    }

    train_snac(
        env_name=args.env_name,
        experiment_name=args.exp_name,
        agent_params=agent_params,
        updates_per_step=args.updates_per_step,
        seed=args.seed,
        time_steps=args.time_steps,
        start_steps=args.start_steps,
        eval_every=args.eval_every,
        computation_device=computation_device,
        save_model=args.save_model,
        exp_dir = args.exp_dir,
        record_video=args.record_video
    )
