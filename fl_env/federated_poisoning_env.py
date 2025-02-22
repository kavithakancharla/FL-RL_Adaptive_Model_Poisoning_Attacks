import gymnasium as gym
from gymnasium import spaces
import numpy as np


class FederatedPoisoningEnv(gym.Env):
    def __init__(self, num_clients=100, num_malicious=10):
        super(FederatedPoisoningEnv, self).__init__()
        self.num_clients = num_clients
        self.num_malicious = num_malicious
        self.current_round = 0

        # Define Action and Observation Space
        self.action_space = spaces.Box(low=-1, high=1, shape=(num_malicious,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_clients,), dtype=np.float32)

        # State Variables
        self.global_model_accuracy = 1.0
        self.state = np.ones(self.num_clients)

    def reset(self):
        self.current_round = 0
        self.global_model_accuracy = 1.0
        self.state = np.ones(self.num_clients)
        return self.state

    def step(self, action):
        # Simulate Poisoning Action
        poison_effect = np.sum(action) / len(action)
        self.global_model_accuracy -= poison_effect

        # Reward Calculation: Negative accuracy as we want to minimize it
        reward = -self.global_model_accuracy

        # Update State
        self.state = np.clip(self.state - poison_effect, 0, 1)

        # Check if training should terminate
        done = self.global_model_accuracy < 0.2 or self.current_round > 100
        self.current_round += 1

        return self.state, reward, done, {}

    def render(self):
        print(f"Round: {self.current_round}, Accuracy: {self.global_model_accuracy}")
