
from functools import partial
from itertools import product
import gymnasium as gym
from tqdm import tqdm
from stable_baselines3 import PPO, DQN


def learn_and_save(model, name, **kwargs):
    model.learn(**kwargs)
    model.save("models/" + name)

time_steps = 1_000_000

algorithms = {
    "ppo": partial(PPO, "MlpPolicy", verbose=1),
    "dqn": partial(DQN, "MlpPolicy", verbose=1),
}

envs = [
    "CartPole-v1",
    "LunarLander-v3",
]

tests = {
    f"{algo}_{env}_{time_steps}": (model(env=gym.make(env)), env)
    for (algo, model), env in 
    product(algorithms.items(), envs)
}


for name, (model, env) in tqdm(tests.items()):
    learn_and_save(model, name, total_timesteps=time_steps)