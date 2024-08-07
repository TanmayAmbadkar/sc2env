import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pysc2.agents import base_agent
from pysc2.env import sc2_env, run_loop
from pysc2.lib import actions, features

# Define the Deep Q-Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)

# Custom SC2 Agent
class Agent(base_agent.BaseAgent):
    def __init__(self):
        super(Agent, self).__init__()
        self.state_dim = 12  # Adjust based on your state features
        self.action_dim = len(smart_actions)  # Set based on the number of possible actions

        self.model = DQN(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.memory = []
        self.batch_size = 32
        self.gamma = 0.9  # Discount factor for future rewards

    def step(self, obs):
        super(Agent, self).step(obs)
        # Implement the logic to select actions based on state observed
        return actions.FUNCTIONS.no_op()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.model(next_states).max(1)[0]
        next_q_values[dones] = 0
        expected_q_values = rewards + self.gamma * next_q_values

        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear some memory if needed
        if len(self.memory) > 10000:
            self.memory = self.memory[-10000:]

# Initialize and run the StarCraft II environment
def main():
    agent = Agent()
    try:
        with sc2_env.SC2Env(
            map_name="Simple64",
            players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=True
        ) as env:
            run_loop.run_loop([agent], env, max_episodes=1000)  # Set your desired max_episodes
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        print("Finished")

if __name__ == "__main__":
    main()
