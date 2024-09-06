from stable_baselines3.common.callbacks import BaseCallback
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

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
            'remaining_enemies': 0
        }
        
        if info['done']:
            self.logger.record("game_stats/win", info['is_success'])
            self.logger.record("game_stats/enemies_killed", info['enemies_killed'])
            self.logger.record("game_stats/allies_killed", info['allies_killed'])
            self.logger.record("game_stats/remaining_allies", info['remaining_allies'])
            self.logger.record("game_stats/remaining_enemies", info['remaining_enemies'])
        
        
        return True
