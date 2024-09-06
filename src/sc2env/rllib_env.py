from sc2env.env import SC2GymWrapper

class SC2GymWrapperRllib(SC2GymWrapper):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, env_config = None):
        super().__init__(env_config['map_name'], env_config['player_race'], env_config['bot_race'],  env_config['bot_difficulty'])
