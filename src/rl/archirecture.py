import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AtariNetExtractor(BaseFeaturesExtractor):
    """
    Custom CNN for processing multiple inputs (screen, minimap, non-spatial features, and available actions).
    
    :param observation_space: (gym.spaces.Dict)
    :param features_dim: (int) Number of features extracted.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super(AtariNetExtractor, self).__init__(observation_space, features_dim)
        
        # CNN for screen input
        n_input_channels_screen = observation_space['screen'].shape[0]
        self.screen_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels_screen, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # CNN for minimap input
        n_input_channels_minimap = observation_space['minimap'].shape[0]
        self.minimap_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels_minimap, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Fully connected layer for non-spatial features
        n_input_non_spatial = observation_space['player_data'].shape[0]
        self.non_spatial_fc = nn.Sequential(
            nn.Linear(n_input_non_spatial, 64),
            nn.Tanh(),
        )
        
        # Fully connected layer for available actions (size will be number of actions)
        n_input_available_actions = observation_space['available_actions'].shape[0]
        self.available_actions_fc = nn.Sequential(
            nn.Linear(n_input_available_actions, 64),
            nn.Tanh(),
        )
        
        # Compute the number of features extracted by CNNs (screen and minimap) and FC layers
        with th.no_grad():
            n_flatten_screen = self.screen_cnn(
                th.as_tensor(observation_space['screen'].sample()[None]).float()
            ).shape[1]
            
            n_flatten_minimap = self.minimap_cnn(
                th.as_tensor(observation_space['minimap'].sample()[None]).float()
            ).shape[1]
        
        # Total size of concatenated features from screen, minimap, non-spatial features, and available actions
        self.total_concat_size = n_flatten_screen + n_flatten_minimap + 64 + 64
        
        # Final linear layer to reduce to desired feature size
        self.linear = nn.Sequential(nn.Linear(self.total_concat_size, features_dim), nn.ReLU())

    def forward(self, observations):
        # Process screen, minimap, and non-spatial inputs
        screen_features = self.screen_cnn(observations['screen'])
        minimap_features = self.minimap_cnn(observations['minimap'])
        non_spatial_features = self.non_spatial_fc(observations['player_data'])
        available_actions_features = self.available_actions_fc(observations['available_actions'])

        screen_features = th.log(screen_features)
        minimap_features = th.log(minimap_features)
        non_spatial_features = th.log(non_spatial_features)
        # Concatenate all features
        concatenated_features = th.cat([screen_features, minimap_features, non_spatial_features, available_actions_features], dim=1)
        
        # Pass through final linear layer
        return self.linear(concatenated_features)
