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
        value_output = torch.tanh(self.value_head(x))
        return policy_logits, value_output

    def save(self, filename, optimizer):
        """ Saves the model state along with optimizer state. """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, filename)
        print(f"Model and optimizer states have been saved to {filename}")

    def load(self, filename, device):
        """ Loads the model and optimizer states. """
        checkpoint = torch.load(filename, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights have been loaded from", filename)
        return checkpoint['optimizer_state_dict']