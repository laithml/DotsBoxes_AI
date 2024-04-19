import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNetwork(nn.Module):
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, 256)
        self.policy_head = nn.Linear(256, 8 * 7 + 7 * 8)  # Total number of possible moves
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        policy_logits = self.policy_head(x)
        policy_output = F.softmax(policy_logits, dim=1)
        value_output = torch.tanh(self.value_head(x))
        return policy_output, value_output

    def save(self, filename="model_checkpoint.pth"):
        """ Saves the model weights to a file. """
        torch.save(self.state_dict(), filename)
        print(f"Model weights have been saved to {filename}")

    def load(self, filename, device='cpu'):
        """ Loads the model weights from a file. """
        self.load_state_dict(torch.load(filename, map_location=device))
        print(f"Model weights have been loaded from {filename}")
