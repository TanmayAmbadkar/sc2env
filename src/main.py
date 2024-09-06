
import gymnasium as gym
from sc2env.env import SC2GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from rl.callbacks import TensorboardCallback
from src.rl.on_policy import learn
from absl import flags


FLAGS = flags.FLAGS
FLAGS([''])


new_logger = configure("./ppo_run/", ["stdout", "csv", "tensorboard"])
env = SC2GymWrapper(map_name="Simple64", player_race="terran", bot_race="random")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log = "./ppo_run/", )


rewards_callback = TensorboardCallback()
learn(model, total_timesteps=60000, log_interval = 1, callback = rewards_callback)


