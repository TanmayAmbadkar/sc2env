
import gymnasium as gym
from sc2env.envs import SC2GymWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from absl import flags


FLAGS = flags.FLAGS
FLAGS([''])

env = SC2GymWrapper(map_name="Simple64", player_race="terran", bot_race="random")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e5))


