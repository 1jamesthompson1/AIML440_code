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
import torch.nn.functional as F

from dsunrise import DSUNRISE 

def train_dsunrise(
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

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if computation_device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
        # Potentially add deterministic flags, though they can impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = gym.wrappers.RescaleAction(gym.make(env_name), min_action=0, max_action=1)
    
    agent = DSUNRISE(
        env=env,
        **agent_params,
        computation_device=computation_device,
    )

    if record_video:
        video_dir = osp.join(exp_dir, "eval_videos")
        os.makedirs(video_dir, exist_ok=True)

    state: np.ndarray = env.reset(seed=seed)[0]
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
    print(f"Running on machine {os.uname().nodename}")
    print(f"Environment name: {env_name}")
    print(f"Start time of training: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(training_start_time))}")
    print(f"Using settings: updates_per_step={updates_per_step}, seed={seed}, time_steps={time_steps}, start_steps={start_steps}, eval_every={eval_every}, exp_dir={exp_dir}, save_model={save_model}, experiment_name={experiment_name}", flush=True)


    for frame_idx in range(1, time_steps + 1):
        episode_timesteps += 1

        action: np.ndarray
        mask: np.ndarray
        action, mask = agent.select_action(state)

        next_state: np.ndarray
        reward: Any
        terminated: bool
        truncated: bool
        next_state, reward, terminated, truncated, _ = env.step(action[0])
        done: bool = terminated or truncated

        # Ensure reward is float
        reward_float = float(reward)

        agent.memory.push(state, action, reward_float, next_state, done, mask)

        state = next_state
        episode_reward += reward_float

        if frame_idx >= start_steps:
            for _ in range(updates_per_step):
                agent.update_parameters()

        if done:
            episode_end_time = time.time()
            episode_duration = episode_end_time - episode_start_time
            print(
                f"Total T: {frame_idx} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f} Duration: {episode_duration:.2f}s",
                 flush=True
            )
            state = env.reset(seed=seed + episode_num + 1)[0]
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1
            episode_start_time = time.time()

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

            # if save_model: # Use save_model parameter
            #     agent.save_model(osp.join(exp_dir, "agent.pt"))

    env.close()
    end_training_time = time.time()
    total_training_duration = end_training_time - training_start_time

    print(
        f"Training completed in {total_training_duration:.2f}s. Which is {total_training_duration/3600} hours.",
    )
    # agent.save_model(osp.join(exp_dir, "agent_final.pt"))



def evaluate_model(env_name: str, agent: DSUNRISE, seed: int, num_episodes: int=10, max_steps: int=1000, video_dir = None):
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
            eval_env = gym.wrappers.RescaleAction(gym.make(env_name, render_mode='rgb_array'), min_action=0, max_action=1)
            eval_env = gym.wrappers.RecordVideo(
                eval_env,
                video_folder=video_dir,
                episode_trigger=lambda ep_id: ep_id == 0,  # Only record the first episode
                name_prefix=f"eval-{seed}"
            )
        else:
            eval_env = gym.wrappers.RescaleAction(gym.make(env_name), min_action=0, max_action=1)
        
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
    

def parse_args():
    parser = argparse.ArgumentParser()
    # architecture
    parser.add_argument(
        "--critic-arch", type=int, nargs='+', default=[256, 256], help="Architecture of the critic networks (list of hidden layer sizes)"
    )
    parser.add_argument(
        "--actor-arch", type=int, nargs='+', default=[256, 256],help="Architecture of the actor networks (list of hidden layer sizes)"
    )
    
    # train
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--save-freq', default=0, type=int)
    parser.add_argument('--computation-device', default='cpu', type=str)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    
    # env
    parser.add_argument('--env', default="Ant-v5", type=str)
    
    # ensemble
    parser.add_argument('--num-ensemble', default=3, type=int)
    parser.add_argument('--ber-mean', default=0.5, type=float)
    
    # inference
    parser.add_argument('--inference-type', default=0.0, type=float)
    parser.add_argument('--feedback-type', default=1, type=int)
    
    # corrective feedback
    parser.add_argument('--temperature', default=20.0, type=float)
    parser.add_argument('--temperature-act', default=0.0, type=float)

    parser.add_argument('--expl-gamma', default=0.0, type=float)

    parser.add_argument('--exp-name', type=str)
    parser.add_argument('--exp-dir', type=str)
    
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    
    # noinspection PyTypeChecker
    variant = dict(
        version="normal",
    )

    computation_device = torch.device(args.computation_device)

    agent_params = dict(
        num_ensemble=args.num_ensemble,
        actor_arch = args.actor_arch,
        critic_arch = args.critic_arch,
        temperature=args.temperature,
        temperature_act=args.temperature_act,
        feedback_type=args.feedback_type,
        # inference_type=args.inference_type,
        actor_lr=3E-4,
        critic_lr=3E-4,
        entropy_lr=3E-4,
        soft_target_tau=5e-3,
        target_update_period=1,
        discount=0.99,
        expl_gamma=args.expl_gamma,
        max_replay_buffer_size=int(1E6),
        batch_size=args.batch_size,
        reward_scale=1.0,
        auto_entropy_tuning=True,

    )
                            
            
    # if 'cuda' in args.computation_device:
    #     ptu.set_gpu_mode(True, gpu_id=args.computation_device[0])
    # else:
    #     ptu.set_gpu_mode(False)
    
    train_dsunrise(
        experiment_name=args.exp_name,
        exp_dir=args.exp_dir,
        env_name=args.env,
        agent_params=agent_params,
        updates_per_step=1,
        seed=args.seed,
        time_steps=int(1E6),
        start_steps=1000,
        eval_every=10_000,
        save_model=True,
        record_video=True,
        computation_device=computation_device,
    )