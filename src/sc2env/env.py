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
    
    def __init__(self, map_name="DefeatZerglingsAndBanelings", player_race=sc2_env.Race.terran, bot_race=sc2_env.Race.zerg, bot_difficulty=sc2_env.Difficulty.hard, reward_shaping = {}, **kwargs):
        super().__init__()
        self.map_name = map_name
        self.player_race = player_race
        self.bot_race = bot_race
        self.bot_difficulty = bot_difficulty
        self.env = None
        self.reward_shaping = reward_shaping
        
        self.init_env()
        
        # Define observation space for Stable-Baselines3
        
        
        # Define the action space (customize based on the actions available in your game)
        # self.action_space = spaces.Discrete(123)  # Adjust as needed based on action dimensions
        action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]

        # some additional actions for minigames (not necessary to solve)
        # action_ids += [11, 71, 72, 73, 74, 79, 140, 168, 239, 261, 264, 269, 274, 318, 335, 336, 453, 477]

        # Set up the action space based on available actions and their argument sizes
        # action_ids = list(actions.FUNCTIONS)
        self.act_wrapper = ActionWrapper(64, action_ids)

        self.action_space = spaces.MultiDiscrete([len(action_ids), 64, 64])
        
        self.observation_space = spaces.Dict({
            
            
            'player_data': spaces.Box(low=0, high=64, shape=(11,), dtype=np.float32),
            'available_actions': spaces.Box(low=0, high=1, shape=(len(action_ids),), dtype=np.int32),
            # Screen feature layers
            'screen': spaces.Box(low=0, high=255, shape=(13, 64, 64), dtype=np.float32),
            
            # Minimap feature layers
            'minimap': spaces.Box(low=0, high=255, shape=(7, 64, 64), dtype=np.float32),
        })
        
        self.action_id_func_map = {action_id: count for count, action_id in enumerate(action_ids)}

        print(self.observation_space)
        print(self.action_space)
        
        # self.act_wrapper = ActionWrapper(64, action_ids)
        

    def init_env(self):
        settings = {
            'map_name': self.map_name,
            'players': [sc2_env.Agent(convert_to_race(self.player_race)),
                        sc2_env.Bot(convert_to_race(self.bot_race), convert_to_difficulty(self.bot_difficulty))],
            'agent_interface_format': features.AgentInterfaceFormat(
                # action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64, 
                feature_dimensions=features.Dimensions(screen=64, minimap=64),
                use_feature_units=True
            ),
            'realtime': False
        }
        self.env = sc2_env.SC2Env(**settings)
        raw_obs = self.env.reset()[0]

    def reset(self, seed=None, options=None):
        if self.env is None:
            self.init_env()
        
        self.units = []
        raw_obs = self.env.reset()[0]  # Get the raw observation
        self.previous_allies = self.get_units(raw_obs, features.PlayerRelative.SELF)
        self.previous_enemies = self.get_units(raw_obs, features.PlayerRelative.ENEMY)
        self.last_score = raw_obs.observation['score_cumulative'][0]
        
        self.enemies_killed = 0
        self.allies_killed = 0
        # Return the structured observation
        return self.get_derived_obs(raw_obs), {}

    def get_derived_obs(self, raw_obs):
        """
        Converts the raw observation from the SC2 environment into a structured observation
        with three keys: non-spatial features, screen, and minimap.
        """
        # Extract non-spatial features from the raw observation
        non_spatial = self.get_non_spatial_features(raw_obs)
        
        # Extract screen features
        screen_features = raw_obs.observation['feature_screen']
        screen = {
            'height_map': screen_features[features.SCREEN_FEATURES.height_map.index],       # Terrain levels on screen
            'visibility': screen_features[features.SCREEN_FEATURES.visibility_map.index],   # Visibility on screen
            'creep': screen_features[features.SCREEN_FEATURES.creep.index],                 # Creep on screen (Zerg)
            'power': screen_features[features.SCREEN_FEATURES.power.index],                 # Protoss power fields
            'player_id': screen_features[features.SCREEN_FEATURES.player_id.index],         # Unit ownership by player id
            'player_relative': screen_features[features.SCREEN_FEATURES.player_relative.index],  # Friend vs enemy
            'unit_type': screen_features[features.SCREEN_FEATURES.unit_type.index],         # Unit type ID
            'selected': screen_features[features.SCREEN_FEATURES.selected.index],           # Selected units
            'hit_points': screen_features[features.SCREEN_FEATURES.unit_hit_points.index],       # Hit points
            'energy': screen_features[features.SCREEN_FEATURES.unit_energy.index],               # Energy (e.g., for spellcasters)
            'shields': screen_features[features.SCREEN_FEATURES.unit_shields.index],             # Shields (for Protoss units)
            'unit_density': screen_features[features.SCREEN_FEATURES.unit_density.index],   # Unit density in pixels
            'unit_density_aa': screen_features[features.SCREEN_FEATURES.unit_density_aa.index]  # Anti-aliased unit density
        }
        
        screen = np.stack(list(screen.values()))
        # Extract minimap features
        minimap_features = raw_obs.observation['feature_minimap']
        minimap = {
            'height_map': minimap_features[features.MINIMAP_FEATURES.height_map.index],     # Terrain levels on minimap
            'visibility': minimap_features[features.MINIMAP_FEATURES.visibility_map.index], # Visibility on minimap
            'creep': minimap_features[features.MINIMAP_FEATURES.creep.index],               # Creep on minimap (Zerg)
            'camera': minimap_features[features.MINIMAP_FEATURES.camera.index],             # Camera position on minimap
            'player_id': minimap_features[features.MINIMAP_FEATURES.player_id.index],       # Player ownership on minimap
            'player_relative': minimap_features[features.MINIMAP_FEATURES.player_relative.index],  # Friend vs enemy on minimap
            'selected': minimap_features[features.MINIMAP_FEATURES.selected.index]          # Selected units on minimap
        }
        
        minimap = np.stack(list(minimap.values()))
        # Return a dictionary with the 3 keys as the structured observation
        return {
            'player_data': non_spatial['player_data'],
            'available_actions': non_spatial['available_actions'],
            'screen': screen,
            'minimap': minimap
        }
    
    def save_replay(self, replay_dir, prefix):
        if self.env is not None:
            self.env.save_replay(replay_dir, prefix)
            
    def get_non_spatial_features(self, raw_obs):
        """
        Extracts non-spatial features from the raw observation and ensures a consistent observation space
        by using fixed-size arrays and padding/truncating variable-length data.
        """
        
        # Maximum sizes for variable-length features
        # Player general information (Fixed size array)
        player_info = raw_obs.observation["player"]
        player_features = np.array([
            player_info[0],   # player_id
            player_info[1],   # minerals
            player_info[2],   # vespene
            player_info[3],   # food_used
            player_info[4],   # food_cap
            player_info[5],   # food_army
            player_info[6],   # food_workers
            player_info[7],   # idle_worker_count
            player_info[8],   # army_count
            player_info[9] if len(player_info) > 9 else 0,  # warp_gate_count (Protoss)
            player_info[10] if len(player_info) > 10 else 0  # larva_count (Zerg)
        ], dtype=np.float32)
        
        # Available actions (Padded with 0s or truncated to MAX_AVAILABLE_ACTIONS)
        available_actions = np.zeros(len(self.action_id_func_map), dtype=np.int32)
        try:
            actual_available_actions = raw_obs.observation["available_actions"]
        except:
            actual_available_actions = []
        for i in range(len(actual_available_actions)):
            
            if actual_available_actions[i] in self.action_id_func_map:         
                available_actions[self.action_id_func_map[actual_available_actions[i]]] = 1
        # Combine all non-spatial features into a dictionary
        non_spatial_features = {
            'player_data': player_features,        # Fixed-size player information
            'available_actions': available_actions,  # Padded/truncated available actions
        }

        self.mask = np.concatenate([non_spatial_features['available_actions'], np.ones(64), np.ones(64)])
        
        return non_spatial_features

    def valid_action_mask(self):
        
        return self.mask

    def step(self, action):
        raw_obs, rew_valid = self.take_action(action)
    
        reward = self.reward_computation(raw_obs, rew_valid)
        self.last_score = raw_obs.observation['score_cumulative'][0]
        obs = self.get_derived_obs(raw_obs)  # Use the structured observation
        done = raw_obs.last()

        # Calculate important statistics
        current_allies = self.get_units(raw_obs, features.PlayerRelative.SELF)
        current_enemies = self.get_units(raw_obs, features.PlayerRelative.ENEMY)

        allies_killed = len(self.previous_allies) - len(current_allies)
        enemies_killed = len(self.previous_enemies) - len(current_enemies)

        # if done:
        reward+= (enemies_killed - self.enemies_killed)/5 - (allies_killed - self.allies_killed)/5
        
        self.enemies_killed = enemies_killed
        self.allies_killed = allies_killed
        
        # Info dictionary containing the statistics
        info = {
            "done": done,
            "is_success": raw_obs.reward,
            'enemies_killed': enemies_killed,
            'allies_killed': allies_killed,
            'remaining_allies': len(current_allies),
            'remaining_enemies': len(current_enemies),
            'cumulative_score': raw_obs.observation['score_cumulative'][0]
        }
        
        self.info = info

        return obs, reward, done, done, info

    def reward_computation(self, raw_obs, rew_valid):
        
        reward = (raw_obs.observation['score_cumulative'][0] - self.last_score) + rew_valid
        
        for key, value in self.reward_shaping:
            reward += value*raw_obs.observation[key]
            
        if raw_obs.last():
            reward += raw_obs.reward*10 if raw_obs.reward != 0 else 5
        
        return reward
        

    def take_action(self, action):
        
        action_mapped, reward = self.act_wrapper(action, self.valid_action_mask())
        raw_obs = self.env.step(action_mapped)[0]
        return raw_obs, reward

    def get_units(self, obs, alliance=features.PlayerRelative.SELF):
        # Generic method to fetch units of interest based on alliance
        return [unit for unit in obs.observation.raw_units if unit.alliance == alliance]

    def close(self):
        if self.env is not None:
            self.env.close()
        super().close()

    def render(self, mode='human', close=False):
        pass



class ActionWrapper:
    def __init__(self, spatial_dim, action_ids, args=None):
        self.spec = None
        if not args:
            args = [
                'screen',
                'minimap',
                'screen2',
                'queued',
                'control_group_act',
                'control_group_id',
                'select_add',
                'select_point_act',
                'select_unit_act',
                'select_unit_id'
                'select_worker',
                'build_queue_id',
                'unload_id'
            ]
        self.func_ids = action_ids
        self.args, self.spatial_dim = args, spatial_dim

    def __call__(self, action, valid_action_mask):
        defaults = {
            'control_group_act': 0,
            'control_group_id': 0,
            'select_point_act': 0,
            'select_unit_act': 0,
            'select_unit_id': 0,
            'build_queue_id': 0,
            'unload_id': 0,
        }
        
        fn_id_idx = action[0]
        args = []
        
        reward = 0
        valid_action_mask = valid_action_mask[:len(self.func_ids)].astype(int)
        if not valid_action_mask[fn_id_idx]:
            print("Invalid Action:", fn_id_idx, valid_action_mask)
            action[0] = np.random.choice(np.array(self.func_ids)[valid_action_mask])
            reward = -1000
            fn_id_idx = action[0]
        
        # else:
        #     print("Valid Action:", fn_id_idx, valid_action_mask)
        
        fn_id = self.func_ids[fn_id_idx]
        for arg_type in actions.FUNCTIONS[fn_id].args:
            arg_name = arg_type.name
            
            arg = [action[1], action[2]]
            # pysc2 expects all args in their separate lists
            if type(arg) not in [list, tuple]:
                arg = [arg]
            # pysc2 expects spatial coords, but we have flattened => attempt to fix
            if len(arg_type.sizes) == 1 and len(arg) > 1:
                arg = [arg[0]*arg[1] % len(arg_type.sizes)]
            args.append(arg)
            

        return [actions.FunctionCall(fn_id, args)], reward