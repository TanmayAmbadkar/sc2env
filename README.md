
# SC2GymWrapper

**SC2GymWrapper** is a custom Gymnasium environment that wraps the PySC2 environment, providing a flexible interface to run StarCraft II scenarios with Gymnasium-compatible reinforcement learning frameworks. The environment supports dynamic configurations for map, race selection, and game difficulty, and outputs important game statistics such as win/loss status and units killed. Agents can be trained using stable-baselines3.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Parameters](#parameters)
- [Environment Details](#environment-details)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

SC2GymWrapper allows users to run StarCraft II scenarios as Gymnasium environments. It is designed for ease of integration with reinforcement learning algorithms, offering a seamless interface for interacting with the PySC2 environment while providing dynamic control over game settings.

## Features

- **Flexible Race Selection:** Use string parameters to select races dynamically (`"terran"`, `"zerg"`, `"protoss"`, `"random"`).
- **Dynamic Observation Space:** Adjusts to the number of units in the game, providing relevant attributes (position, health).
- **Comprehensive Statistics:** Outputs game statistics such as win/loss status, number of enemies killed, and number of allies killed.
- **Compatibility with Gymnasium:** Easily integrates with Gymnasium-compatible reinforcement learning frameworks.

## Installation

1. **Clone the Repository:**
   
2. **Install Dependencies:**
   Make sure you have Python 3.8 or above installed. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Install StarCraft II:**
   Download and install StarCraft II from Blizzard and ensure it is accessible on your system.

4. **Install PySC2 Maps:**
   Download the necessary PySC2 maps and place them in the appropriate directory as specified in the PySC2 documentation.

## Usage

Here’s a basic example to get started with SC2GymWrapper and stable-baselines3:

```python

import gymnasium as gym
from sc2env.envs import SC2GymWrapper
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from absl import flags


FLAGS = flags.FLAGS
FLAGS([''])

env = SC2GymWrapper(map_name="Simple64", player_race="terran", bot_race="random")
# use ppo2 to learn and save the model when finished
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=int(1e5))



```

## Parameters

- `map_name` (str): The name of the StarCraft II map to play.
- `player_race` (str): The race of the player agent (`"terran"`, `"zerg"`, `"protoss"`, `"random"`).
- `bot_race` (str): The race of the bot opponent (`"terran"`, `"zerg"`, `"protoss"`, `"random"`).
- `bot_difficulty` (sc2_env.Difficulty): The difficulty of the bot opponent (e.g., `sc2_env.Difficulty.hard`).

## Environment Details

- **Action Space:** Discrete actions mapped to unit movements and attacks.
- **Observation Space:** A dynamic array capturing the position `(x, y)` and health of each unit up to the specified `max_units`.
- **Info Dictionary:** Contains important statistics like `is_success`, `enemies_killed`, `allies_killed`, `remaining_allies`, and `remaining_enemies`.
