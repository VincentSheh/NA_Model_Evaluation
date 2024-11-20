import numpy as np
import pandas as pd
import random

class QLearningAgent:
    def __init__(self, vim, cpu_range=(0.5, 5.5), step_size=0.5, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995, exploration_commit_steps=1000):
        self.vim = vim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_commit_steps = exploration_commit_steps
        # Generate a discrete action space with a step size of 0.5
        self.cpu_range = np.arange(cpu_range[0], cpu_range[1] + step_size, step_size)
        self.num_actions = len(self.cpu_range)  # Update number of actions based on cpu_range
        self.q_table = {}  # Q-table initialized as an empty dictionary

    def get_state(self, cur_info):
        """Convert current information into a tuple state for Q-table."""
        return (
            # cur_info["qoe"],
            cur_info["n_streamers"],
            round(cur_info["ori_intensity"]/200)*200,
            # cur_info["red_intensity"],
            # cur_info["video_cpu"]
        )

    def choose_action(self, state):
        """Choose action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.cpu_range)  # Exploration: Random action
        else:
            # Exploitation: Choose the best action for the current state
            q_values = self.q_table.get(state, np.zeros(self.num_actions))
            return self.cpu_range[np.argmax(q_values)]

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table using the Q-learning formula."""
        
        # Find the closest action in cpu_range
        action_index = (np.abs(self.cpu_range - action)).argmin()
        
        # Get the current Q-value for the (state, action) pair
        current_q = self.q_table.get(state, np.zeros(self.num_actions))[action_index]
        
        # Max Q-value for the next state
        next_max_q = max(self.q_table.get(next_state, np.zeros(self.num_actions)))
        
        # Q-learning formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_max_q - current_q)
        
        # Update the Q-table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        self.q_table[state][action_index] = new_q
        
        # Decay exploration rate
        
        if self.vim.area.current_timestep > self.exploration_commit_steps:
            self.exploration_rate *= self.exploration_decay

    def take_action(self, cur_info):
        """Take an action and update CPU allocation."""
        state = self.get_state(cur_info)
        action = self.choose_action(state)
        video_cpu = action
        
        return video_cpu