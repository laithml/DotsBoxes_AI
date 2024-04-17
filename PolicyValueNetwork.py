import torch
import torch.nn as nn


class PolicyValueNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.policy_head = nn.Linear(64, output_size)  # Output layer for policy
        self.value_head = nn.Linear(64, 1)  # Output layer for value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = torch.softmax(self.policy_head(x), dim=-1)  # Policy output
        value = self.value_head(x)  # Value output
        return policy, value
