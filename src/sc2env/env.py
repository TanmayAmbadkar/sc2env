import gymnasium as gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gymnasium import spaces
import logging
import numpy as np
import sys

logger = logging.getLogger(__name__)

def convert_to_race(race_str):
    """
    Converts a string to the corresponding sc2_env.Race value using getattr.
    Defaults to sc2_env.Race.random if the race name is not found.
    """
    try:
        # Dynamically get the Race attribute using getattr
        return getattr(sc2_env.Race, race_str.lower())
    except AttributeError:
        # Default to 'random' if the race name is invalid
        logger.warning(f"Invalid race name '{race_str}', defaulting to 'random'.")
        return sc2_env.Race.random
    
def convert_to_difficulty(difficulty_str):
    """
    Converts a string to the corresponding sc2_env.difficulty value using getattr.
    Defaults to sc2_env.Difficulty.easy if the race name is not found.
    """
    try:
        # Dynamically get the Race attribute using getattr
        return getattr(sc2_env.Difficulty, difficulty_str.lower())
    except AttributeError:
        # Default to 'random' if the race name is invalid
        logger.warning(f"Invalid difficulty name '{difficulty_str}', defaulting to 'random'.")
        return sc2_env.Difficulty.easy

class SC2GymWrapper(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, map_name="DefeatZerglingsAndBanelings", player_race=sc2_env.Race.terran, bot_race=sc2_env.Race.zerg, bot_difficulty=sc2_env.Difficulty.hard, **kwargs):
        super().__init__()
        self.map_name = map_name
        self.player_race = player_race
        self.bot_race = bot_race
        self.bot_difficulty = bot_difficulty
        self.env = None
        
        
        self.init_env()
        
        # Placeholder for units
        self.units = []
        self.previous_allies = []
        self.previous_enemies = []
        
        # Define action space
        self.action_space = spaces.Discrete(123)  # Update based on map needs
        
        # Define observation space dynamically
        self.observation_space = spaces.Box(
            low=0, 
            high=64, 
            shape=(self.max_units * self.action_len,),  # Each unit's (x, y, hp)
            dtype=np.uint8
        )
        

    def init_env(self):
        settings = {
            'map_name': self.map_name,
            'players': [sc2_env.Agent(convert_to_race(self.player_race)),
                        sc2_env.Bot(convert_to_race(self.bot_race), convert_to_difficulty(self.bot_difficulty))],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64),
            'realtime': False
        }
        self.env = sc2_env.SC2Env(**settings)
        raw_obs = self.env.reset()[0]
        self.max_units = len(self.get_units(raw_obs))
        self.action_len = len(self.get_units(raw_obs)[0])
        

    def reset(self, seed=None):
        if self.env is None:
            self.init_env()
        
        self.units = []
        raw_obs = self.env.reset()[0]
        self.previous_allies = self.get_units(raw_obs, features.PlayerRelative.SELF)
        self.previous_enemies = self.get_units(raw_obs, features.PlayerRelative.ENEMY)
        return self.get_derived_obs(raw_obs), {}

    def get_derived_obs(self, raw_obs):
        # Extract units dynamically based on the raw observation.
        self.units = self.get_units(raw_obs)
        obs = np.zeros((self.max_units, self.action_len), dtype=np.uint8)
        
        for i, unit in enumerate(self.units):
            
            obs[i] = np.array(unit)
        
        return obs.flatten()

    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward + raw_obs.observation['score_cumulative'][0]
        obs = self.get_derived_obs(raw_obs)
        done = raw_obs.last()

        # Calculate important statistics
        current_allies = self.get_units(raw_obs, features.PlayerRelative.SELF)
        current_enemies = self.get_units(raw_obs, features.PlayerRelative.ENEMY)

        allies_killed = len(self.previous_allies) - len(current_allies)
        enemies_killed = len(self.previous_enemies) - len(current_enemies)

        # Update previous unit lists for next step comparison
        self.previous_allies = current_allies
        self.previous_enemies = current_enemies

        # Determine win/loss based on game status
        win = True if raw_obs.reward > 0 else False
        loss = 1 if done and raw_obs.reward <= 0 else 0

        # Info dictionary containing the statistics
        info = {
            "done": done,
            "is_success" : win,
            'enemies_killed': enemies_killed,
            'allies_killed': allies_killed,
            'remaining_allies': len(current_allies),
            'remaining_enemies': len(current_enemies),
            'cumulative_score': raw_obs.observation['score_cumulative'][0]
        }
        
        self.info = info

        return obs, reward, done, done, info

    def take_action(self, action):
        # Map actions dynamically based on context.
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        elif action <= 32:
            direction = np.floor((action - 1) / 8)
            idx = (action - 1) % 8
            if direction == 0:
                action_mapped = self.move_unit(idx, "up")
            elif direction == 1:
                action_mapped = self.move_unit(idx, "down")
            elif direction == 2:
                action_mapped = self.move_unit(idx, "left")
            else:
                action_mapped = self.move_unit(idx, "right")
        else:
            attacker_idx = np.floor((action - 33) / 9)
            target_idx = (action - 33) % 9
            action_mapped = self.attack_unit(attacker_idx, target_idx)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs

    def move_unit(self, idx, direction):
        if idx >= len(self.units):
            return actions.RAW_FUNCTIONS.no_op()

        unit = self.units[int(idx)]
        if direction == "up":
            new_pos = [unit.x, unit.y - 2]
        elif direction == "down":
            new_pos = [unit.x, unit.y + 2]
        elif direction == "left":
            new_pos = [unit.x - 2, unit.y]
        else:
            new_pos = [unit.x + 2, unit.y]

        return actions.RAW_FUNCTIONS.Move_pt("now", unit.tag, new_pos)

    def attack_unit(self, attacker_idx, target_idx):
        if attacker_idx >= len(self.units) or target_idx >= len(self.units):
            return actions.RAW_FUNCTIONS.no_op()

        attacker = self.units[int(attacker_idx)]
        target = self.units[int(target_idx)]
        return actions.RAW_FUNCTIONS.Attack_unit("now", attacker.tag, target.tag)

    def get_units(self, obs, alliance=features.PlayerRelative.SELF):
        # Generic method to fetch units of interest based on alliance
        return [unit for unit in obs.observation.raw_units if unit.alliance == alliance]

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass
