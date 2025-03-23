
from functools import partial
from itertools import product
import gymnasium as gym
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def learn_and_save(model, name, **kwargs):
    model.learn(**kwargs)
    # model.save("models/" + name)

from stable_baselines3 import PPO, TD3, SAC

time_steps = 1_000_000

tensorboard_logs = "logs/cont_model_training"
algorithms = {
    "sac": partial(SAC, "MlpPolicy", verbose=0, tensorboard_log = tensorboard_logs),
    "ppo": partial(PPO, "MlpPolicy", verbose=0, tensorboard_log = tensorboard_logs),
    "td3": partial(TD3, "MlpPolicy", verbose=0, tensorboard_log = tensorboard_logs),
}

envs = [
    "HalfCheetah-v5",
    "Walker2d-v5",
    "Humanoid-v5",
]

continuous_tests = {
    f"{algo}_{env}_{time_steps}": (model(env=gym.make(env)), env)
    for (algo, model), env in 
    product(algorithms.items(), envs)
}


with ThreadPoolExecutor() as executor:
    futures = [executor.submit(learn_and_save, model, name, total_timesteps=time_steps) for name, (model, _) in continuous_tests.items()]
    for future in tqdm(futures):
        future.result()