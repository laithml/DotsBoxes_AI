import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.policy_head = nn.Linear(256, 8 * 7 + 7 * 8)  # Total number of possible moves
        # Value output for estimating the game outcome
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        #TODO: skip connection resNet
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        # Policy head outputs a probability distribution over moves
        policy_logits = self.policy_head(x)
        policy_output = F.softmax(policy_logits, dim=1)

        # Value head predicts the expected outcome using tanh
        value_output = torch.tanh(self.value_head(x))

        return policy_output, value_output
