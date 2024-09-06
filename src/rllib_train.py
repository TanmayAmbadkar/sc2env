import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from sc2env.rllib_env import SC2GymWrapperRllib
from rl.sb3_utils import learn
from absl import flags
import sys



if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS(sys.argv)

    env_config = {'map_name': "Simple64", "player_race": "terran", "bot_race": "random", "bot_difficulty":"easy"}
    # env = SC2GymWrapper(map_name="Simple64", player_race="terran", bot_race="random")
    ray.init()
    algo = ppo.PPO(env=SC2GymWrapperRllib, config={
        "env_config": env_config,  # config to pass to env class
    })

    while True:
        print(algo.train())