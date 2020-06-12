"""
The experience replay class was taken from
https://edersantana.github.io/articles/keras_rl/
This version has been modified to use PyTorch instead of Keras and numpy
"""
import torch
import numpy as np


class ExperienceReplay(object):
    def __init__(self, num_actions, max_memory=100, gamma=0.9, init_memory=list()):
        self.max_memory = max_memory
        self.memory = init_memory
        self.discount = gamma
        self.num_actions = num_actions

    def remember(self, states, is_crashed):
        # memory[i] = [[state_t, action_t, reward_t, state_t1], game_over?]
        self.memory.append([states, is_crashed])
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)

    def get_batch(self, model, device, batch_size=10):  # Make Batch Size global
        len_memory = len(self.memory)
        state_dim = self.memory[0][0][0].shape[0]  # Should be equal to 5
        inputs = torch.zeros(min(len_memory, batch_size), state_dim).to(device)  # Should be shape(100, 5)
        targets = torch.zeros(inputs.shape[0], self.num_actions).to(device)  # Should be shape(100, 2)
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            # Randomly select indices of experiences from memory
            state_t, action_t, reward_t, state_t1 = self.memory[idx][0]
            is_crashed = self.memory[idx][1]

            inputs[i] = state_t
            targets[i] = model(state_t.to(device))
            q_sa = torch.max(model(state_t1.to(device)))  # the model will directly output a list of Q values
            if is_crashed:  # Future rewards don't exist so Q value does not increase. This discourages crashing
                targets[i, action_t] = reward_t
            else:
                # Bellman Equation
                targets[i, action_t] = reward_t + self.discount * q_sa
        return inputs, targets

    def get_memory(self):
        return self.memory

    def load_memory(self, memory):
        self.memory = memory

    def get_batch_optim(self, model, device, batch_size=10):
        len_memory = len(self.memory)
        state_dim = self.memory[0][0][0].shape[0]

        inputs = torch.zeros(min(len_memory, batch_size), state_dim).to(device)
        changed_states = torch.zeros(inputs.shape[0], state_dim)
        crashed_states = np.zeros(inputs.shape[0])
        targets = torch.zeros(inputs.shape[0], self.num_actions).to(device)
        actions = np.zeros(inputs.shape[0], dtype=int)
        rewards = np.zeros(inputs.shape[0])

        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            # Randomly select indices of experiences from memory
            state_t, action_t, reward_t, state_t1 = self.memory[idx][0]
            is_crashed = self.memory[idx][1]

            inputs[i, :] = state_t
            actions[i] = action_t
            rewards[i] = reward_t
            changed_states[i, :] = state_t1
            crashed_states[i] = is_crashed

        targets = model(inputs)
        targets = targets.cpu().detach().numpy()

        q_sa = torch.max(model(changed_states), 1)
        q_sa = q_sa[0].cpu().detach().numpy()

        crashed = np.where(crashed_states == True)[0]
        non_crashed = np.where(crashed_states == False)[0]

        targets[crashed, actions[crashed]] = rewards[crashed]
        targets[non_crashed, actions[non_crashed]] = rewards[non_crashed] + self.discount * q_sa[non_crashed]

        return inputs, torch.from_numpy(targets)
