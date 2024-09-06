
import gymnasium as gym
from sc2env.env import SC2GymWrapper
from stable_baselines3 import PPO
from rl.sb3_utils import learn
from absl import flags


FLAGS = flags.FLAGS
FLAGS([''])

env = SC2GymWrapper(map_name="Simple64", player_race="terran", bot_race="random")
model = PPO("MlpPolicy", env, verbose=1)
learn(model, total_timesteps=int(1e5))


