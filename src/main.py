import json
from sc2env.env import SC2GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from rl.callbacks import TensorboardCallback
from rl.on_policy import learn
from absl import flags, app

# Load configuration from JSON file
def load_config(config_file="config.json"):
    with open(config_file, "r") as file:
        config = json.load(file)
    return config

def main(argv):
    # Load the config variables
    config = load_config()

    # Extract variables from the config
    map_name = config["map_name"]
    player_race = config["player_race"]
    bot_race = config["bot_race"]
    total_timesteps = config["total_timesteps"]
    log_name = config["log_name"]

    # Configure logger
    new_logger = configure("./ppo_run/", ["stdout", "csv", "tensorboard"])

    # Create the environment using the config parameters
    env = SC2GymWrapper(map_name=map_name, player_race=player_race, bot_race=bot_race)
    
    state, info = env.reset()
    print(state)

    # # Initialize the PPO model
    # model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_run/")

    # # Define the callback for rewards
    # rewards_callback = TensorboardCallback()

    # # Start the learning process with the specified total timesteps
    # learn(
    #     model,
    #     total_timesteps=total_timesteps,
    #     log_interval=1,
    #     callback=rewards_callback,
    #     tb_log_name=log_name,
    # )

if __name__ == "__main__":
    # Run the main function
    app.run(main)
