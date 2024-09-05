
import gymnasium as gym
from sc2env.envs import DZBEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from absl import flags


FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
# env = gym.make('defeat-zerglings-banelings-v0')
eng = DZBEnv()
check_env(eng)
# env = DummyVecEnv([lambda: DZBEnv()])

# use ppo2 to learn and save the model when finished
model = PPO("MlpPolicy", eng, verbose=1)
model.learn(total_timesteps=int(1e5))
# model.save("model/dbz_ppo")