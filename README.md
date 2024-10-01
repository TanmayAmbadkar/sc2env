
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
- **Dict Observation Space:** Uses minimap, screen and non-spatial features as observation space
- **Multidiscrete Action Space:** Can take multi-discrete actions for function(args) based action in pysc2
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

4. **Modify sb3-contrib for maskable PPO:**
   Modify MaskableActorCriticPolicy forward() by adding 
   ```python
   elif "available_actions" in obs:
      action_masks = th.cat([obs['available_actions'].reshape(-1, ), th.ones(64).to(obs['available_actions'].device), th.ones(64).to(obs['available_actions'].device)])
      distribution.apply_masking(action_masks)
   ```
   at line 140

## Usage

Modify config to add map, player race, bot race and total_timesteps
