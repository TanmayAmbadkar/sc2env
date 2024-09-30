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
        
        # Define observation space for Stable-Baselines3
        self.observation_space = spaces.Dict({
            # Non-spatial features (General player information, control groups, etc.)
            # 'non_spatial': spaces.Dict({
            #     'player_data': spaces.Box(low=0, high=np.inf, shape=(11,), dtype=np.float32),  # Player info
            #     # 'control_groups': spaces.Box(low=0, high=np.inf, shape=(10, 2), dtype=np.float32),  # Control groups
            #     'available_actions': spaces.Box(low=0, high=np.inf, shape=(500,), dtype=np.int32),  # Available actions
            #     # 'last_actions': spaces.Box(low=0, high=np.inf, shape=(10,), dtype=np.int32),  # Last actions
            #     # 'action_results': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),  # Action result
            #     # 'alerts': spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.int32),  # Alerts
            # }),
            
            'player_data': spaces.Box(low=0, high=np.inf, shape=(11,), dtype=np.float32),
            'available_actions': spaces.Box(low=0, high=np.inf, shape=(500,), dtype=np.int32),
            # Screen feature layers
            'screen': spaces.Box(low=0, high=np.inf, shape=(13, 64, 64), dtype=np.float32),
            #     spaces.Dict({
            #     'height_map': spaces.Box(low=0, high=255, shape=(64, 64), dtype=np.uint8),  # Terrain height on screen
            #     'visibility': spaces.Box(low=0, high=3, shape=(64, 64), dtype=np.uint8),  # Visibility on screen (hidden, seen, visible)
            #     'creep': spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.uint8),  # Zerg creep presence on screen
            #     'power': spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.uint8),  # Protoss power fields on screen
            #     'player_id': spaces.Box(low=0, high=np.inf, shape=(64, 64), dtype=np.uint8),  # Unit ownership on screen (player IDs)
            #     'player_relative': spaces.Box(low=0, high=4, shape=(64, 64), dtype=np.uint8),  # Friendly vs hostile units on screen
            #     'unit_type': spaces.Box(low=0, high=np.inf, shape=(64, 64), dtype=np.uint16),  # Unit type IDs on screen
            #     'selected': spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.uint8),  # Selected units on screen
            #     'hit_points': spaces.Box(low=0, high=np.inf, shape=(64, 64), dtype=np.uint16),  # Hit points of units on screen
            #     'energy': spaces.Box(low=0, high=np.inf, shape=(64, 64), dtype=np.uint16),  # Energy of units on screen
            #     'shields': spaces.Box(low=0, high=np.inf, shape=(64, 64), dtype=np.uint16),  # Shields of protoss units on screen
            #     'unit_density': spaces.Box(low=0, high=255, shape=(64, 64), dtype=np.uint8),  # Unit density per pixel
            #     'unit_density_aa': spaces.Box(low=0, high=255, shape=(64, 64), dtype=np.uint8)  # Anti-aliased unit density
                
            # }),
            
            # Minimap feature layers
            'minimap': spaces.Box(low=0, high=np.inf, shape=(7, 64, 64), dtype=np.float32),
            # spaces.Dict({
            #     'height_map': spaces.Box(low=0, high=255, shape=(64, 64), dtype=np.uint8),  # Terrain height on minimap
            #     'visibility': spaces.Box(low=0, high=3, shape=(64, 64), dtype=np.uint8),  # Visibility on minimap (hidden, seen, visible)
            #     'creep': spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.uint8),  # Zerg creep on minimap
            #     'camera': spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.uint8),  # Visible area on the minimap (camera location)
            #     'player_id': spaces.Box(low=0, high=np.inf, shape=(64, 64), dtype=np.uint8),  # Unit ownership on minimap (player IDs)
            #     'player_relative': spaces.Box(low=0, high=4, shape=(64, 64), dtype=np.uint8),  # Friendly vs hostile units on minimap
            #     'selected': spaces.Box(low=0, high=1, shape=(64, 64), dtype=np.uint8),  # Selected units on minimap
                
            # })
        })
        
        # Define the action space (customize based on the actions available in your game)
        # self.action_space = spaces.Discrete(123)  # Adjust as needed based on action dimensions
        action_ids = [0, 1, 2, 3, 4, 6, 7, 12, 13, 42, 44, 50, 91, 183, 234, 309, 331, 332, 333, 334, 451, 452, 490]

        # some additional actions for minigames (not necessary to solve)
        action_ids += [11, 71, 72, 73, 74, 79, 140, 168, 239, 261, 264, 269, 274, 318, 335, 336, 453, 477]

        self.action_space = spaces.MultiDiscrete([
            len(action_ids),
            64,
            64
        ])
        
        self.act_wrapper = ActionWrapper(64, action_ids)
        
        print(self.observation_space)

    def init_env(self):
        settings = {
            'map_name': self.map_name,
            'players': [sc2_env.Agent(convert_to_race(self.player_race)),
                        sc2_env.Bot(convert_to_race(self.bot_race), convert_to_difficulty(self.bot_difficulty))],
            'agent_interface_format': features.AgentInterfaceFormat(
                action_space=actions.ActionSpace.RAW,
                use_raw_units=True,
                raw_resolution=64, 
                feature_dimensions=features.Dimensions(screen=64, minimap=64),
                use_feature_units=True
            ),
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
        raw_obs = self.env.reset()[0]  # Get the raw observation
        self.previous_allies = self.get_units(raw_obs, features.PlayerRelative.SELF)
        self.previous_enemies = self.get_units(raw_obs, features.PlayerRelative.ENEMY)
        
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
    def get_non_spatial_features(self, raw_obs):
        """
        Extracts non-spatial features from the raw observation and ensures a consistent observation space
        by using fixed-size arrays and padding/truncating variable-length data.
        """
        
        # Maximum sizes for variable-length features
        MAX_CONTROL_GROUPS = 10        # Max 10 control groups
        MAX_AVAILABLE_ACTIONS = 500    # Example, assume max 500 actions could be available
        MAX_LAST_ACTIONS = 10          # Max number of last actions stored
        MAX_ALERTS = 2                 # Max 2 alerts

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

        # Control Groups (Fixed to MAX_CONTROL_GROUPS, pad or truncate)
        control_groups = np.zeros((MAX_CONTROL_GROUPS, 2), dtype=np.float32)  # (unit_type, count)
        actual_control_groups = raw_obs.observation["control_groups"]
        for i in range(min(len(actual_control_groups), MAX_CONTROL_GROUPS)):
            control_groups[i] = actual_control_groups[i]

        # Available actions (Padded with 0s or truncated to MAX_AVAILABLE_ACTIONS)
        available_actions = np.zeros(MAX_AVAILABLE_ACTIONS, dtype=np.int32)
        try:
            actual_available_actions = raw_obs.observation["available_actions"]
        except:
            actual_available_actions = []
        for i in range(min(len(actual_available_actions), MAX_AVAILABLE_ACTIONS)):
            available_actions[i] = actual_available_actions[i]

        # Last actions (Padded with 0s or truncated to MAX_LAST_ACTIONS)
        last_actions = np.zeros(MAX_LAST_ACTIONS, dtype=np.int32)
        actual_last_actions = raw_obs.observation.get("last_actions", [])
        for i in range(min(len(actual_last_actions), MAX_LAST_ACTIONS)):
            last_actions[i] = actual_last_actions[i]

        # Action results (Padded with 0s or truncated to 1, assuming max 1 result)
        action_result = np.zeros(1, dtype=np.int32)
        actual_action_result = raw_obs.observation.get("action_result", [])
        if len(actual_action_result) > 0:
            action_result[0] = actual_action_result[0]

        # Alerts (Padded with 0s or truncated to MAX_ALERTS)
        alerts = np.zeros(MAX_ALERTS, dtype=np.int32)
        actual_alerts = raw_obs.observation.get("alerts", [])
        for i in range(min(len(actual_alerts), MAX_ALERTS)):
            alerts[i] = actual_alerts[i]

        # Combine all non-spatial features into a dictionary
        non_spatial_features = {
            'player_data': player_features,        # Fixed-size player information
            # 'control_groups': control_groups,      # Padded/truncated control groups
            'available_actions': available_actions,  # Padded/truncated available actions
            # 'last_actions': last_actions,          # Padded/truncated last actions
            # 'action_results': action_result,       # Padded/truncated action result
            # 'alerts': alerts                       # Padded/truncated alerts
        }

        return non_spatial_features


    def step(self, action):
        raw_obs = self.take_action(action)
        reward = raw_obs.reward + raw_obs.observation['score_cumulative'][0]
        obs = self.get_derived_obs(raw_obs)  # Use the structured observation
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
            "is_success": win,
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
        # if action == 0:
        #     action_mapped = actions.RAW_FUNCTIONS.no_op()
        # elif action <= 32:
        #     direction = np.floor((action - 1) / 8)
        #     idx = (action - 1) % 8
        #     if direction == 0:
        #         action_mapped = self.move_unit(idx, "up")
        #     elif direction == 1:
        #         action_mapped = self.move_unit(idx, "down")
        #     elif direction == 2:
        #         action_mapped = self.move_unit(idx, "left")
        #     else:
        #         action_mapped = self.move_unit(idx, "right")
        # else:
        #     attacker_idx = np.floor((action - 33) / 9)
        #     target_idx = (action - 33) % 9
        #     action_mapped = self.attack_unit(attacker_idx, target_idx)

        
        action_mapped = self.act_wrapper(action)
        raw_obs = self.env.step(action_mapped)[0]
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

    def __call__(self, action):
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
        
        fn_id = self.func_ids[fn_id_idx]
        for arg_type in actions.FUNCTIONS[fn_id].args:
            arg_name = arg_type.name
            if arg_name in self.args:
                arg = [action[1], action[2]]
                # pysc2 expects all args in their separate lists
                if type(arg) not in [list, tuple]:
                    arg = [arg]
                # pysc2 expects spatial coords, but we have flattened => attempt to fix
                if len(arg_type.sizes) == 1 and len(arg) > 1:
                    arg = [arg[0] % len(arg_type.sizes)]
                args.append(arg)
            else:
                args.append([defaults[arg_name]])

        return [actions.FunctionCall(fn_id, args)]