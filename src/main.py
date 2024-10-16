import json
from sc2env.env import SC2GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure

from rl.callbacks import TensorboardCallback
from rl.on_policy import learn
from rl.archirecture import AtariNetExtractor
from absl import flags, app
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

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
    use_mask = config['use_mask']
    log_name = f"ppo_{map_name}_{total_timesteps}"

    # Configure logger
    new_logger = configure("./ppo_run/", ["stdout", "csv", "tensorboard"])

    # Create the environment using the config parameters
    env = SC2GymWrapper(map_name=map_name, player_race=player_race, bot_race=bot_race)
        
        
    # Define the policy network architecture
    policy_kwargs = dict(
        features_extractor_class=AtariNetExtractor,
        features_extractor_kwargs=dict(features_dim=256),  # Set to desired feature dimension
        net_arch=dict(pi=[128, 128], vf=[128, 128])  # Two hidden layers of 128 units each for policy and value networks
    )

    if use_mask:
        def mask_fn(env):
            return env.valid_action_mask()


        env = ActionMasker(env, mask_fn)  # Wrap to enable masking

        model = MaskablePPO(
            MaskableActorCriticPolicy, 
            env, 
            policy_kwargs=policy_kwargs,
            verbose=1, 
            tensorboard_log="./ppo_run/",
            learning_rate=0.00001,
            n_epochs=8
        )
    
    # Create the PPO model with the custom extractor and policy
    else:
        model = PPO(
            policy="MultiInputPolicy",  # Use MultiInputPolicy to handle Dict observation space
            env=env,  # Your custom environment here
            policy_kwargs=policy_kwargs,
            verbose=1, tensorboard_log="./ppo_run/",
            learning_rate=0.00001
        )

    # Define the callback for rewards
    rewards_callback = TensorboardCallback(log_name = log_name)

    # Start the learning process with the specified total timesteps
    learn(
        model,
        total_timesteps=total_timesteps,
        log_interval=1,
        callback=rewards_callback,
        tb_log_name=log_name,
        use_masking = use_mask
    )
    
    
    model.save("models/" + log_name)
    
    

if __name__ == "__main__":
    # Run the main function
    app.run(main)
