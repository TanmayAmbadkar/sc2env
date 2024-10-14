from stable_baselines3.common.callbacks import BaseCallback
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, log_name=None):
        super().__init__(verbose)
        self.log_name = log_name
        self.update_number = 0

    def _on_step(self) -> bool:
        try:
            info = self.training_env.get_attr("info")[0]
        except Exception as e:
            info = {
            "done": False,
            "is_success" : 0,
            'enemies_killed': 0,
            'allies_killed': 0,
            'remaining_allies': 0,
            'remaining_enemies': 0,
            'cumulative_score': 0
        }
        
        if info['done']:
            self.logger.record("game_stats/win", info['is_success'])
            self.logger.record("game_stats/enemies_killed", info['enemies_killed'])
            self.logger.record("game_stats/allies_killed", info['allies_killed'])
            self.logger.record("game_stats/remaining_allies", info['remaining_allies'])
            self.logger.record("game_stats/remaining_enemies", info['remaining_enemies'])
            self.logger.record("game_stats/cumulative_score", info['cumulative_score'])
        
        
        return True
    def on_rollout_end(self):
        
        env = self.training_env
        obs = env.reset()
        while True:
            
            action, _states = self.model.predict(obs)
            obs, reward, terminated, info = env.step(action)
            if terminated:
                break
        
        env.env_method("save_replay", f"replays/{self.logger.get_dir()}", f"replay_{self.update_number}")
        self.update_number+=1