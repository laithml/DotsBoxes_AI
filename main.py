import numpy as np
from torch import optim

from DotsBoxes import DotsBoxes
from PolicyValueNetwork import PolicyValueNetwork
from PUCTPlayer import PUCTPlayer
from MCTSPlayer import MCTSPlayer

import torch
import torch.nn as nn

class Network:
    def __init__(self, input_size, output_size, learning_rate=0.001):
        self.policy_value_network = PolicyValueNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_value_network.parameters(), lr=learning_rate)

    def select_action(self, state):
        with torch.no_grad():
            policy, _ = self.policy_value_network(state)
            action_probs = policy.numpy().flatten()
            action = np.random.choice(len(action_probs), p=action_probs)
            return action

    def train(self, states, action_probs, values):
        self.optimizer.zero_grad()
        policy, predicted_values = self.policy_value_network(states)
        policy_loss = -torch.mean(torch.log(policy) * action_probs)
        value_loss = nn.MSELoss()(predicted_values, values)
        total_loss = policy_loss + value_loss
        total_loss.backward()
        self.optimizer.step()

# Assuming you have a function to generate training data from games
def generate_training_data(num_games):
    training_data = []
    for _ in range(num_games):
        game_state = DotsBoxes()
        p1 = PUCTPlayer(game_state, PolicyValueNetwork)
        p2 = MCTSPlayer(game_state)
        while game_state.outcome() == DotsBoxes.ONGOING:
            move = p1.choose_move(100)  # Play with a fixed number of MCTS iterations
            state_tensor = p1.state_to_tensor(game_state)
            policy, value = p1.policy_value_network(state_tensor)
            training_data.append((state_tensor, policy, value))
            game_state.make_move(move[0], move[1], move[2])
            p1, p2 = p2, p1  # Alternate players
    return training_data

# Assuming you have the training data and policy-value network defined
training_data = generate_training_data(1000)

# Assuming you have the PUCT player and human player defined
game_state = DotsBoxes()
p1 = PUCTPlayer(game_state, PolicyValueNetwork)
#
# while game_state.outcome() == DotsBoxes.ONGOING:
#     if game_state.current_player == DotsBoxes.RED:
#         move = p1.choose_move(100)  # Play with a fixed number of MCTS iterations
#     else:
#         orientation = input("Enter line orientation (h for horizontal(---), v for vertical(|)): ").lower()
#         if orientation not in ['h', 'v']:
#             raise ValueError("Invalid orientation")
#         if (orientation == 'h'):
#             i = int(input("Enter row index (0-7): "))
#             j = int(input("Enter column index (0-6): "))
#         else:
#             i = int(input("Enter row index (0-6): "))
#             j = int(input("Enter column index (0-7): "))
#         if not (0 <= i < 8) or not (0 <= j < 8):
#             raise ValueError("Invalid indices")
#         if orientation == 'h' and j == 7:
#             raise ValueError("Invalid row index for horizontal line")
#         if orientation == 'v' and i == 7:
#             raise ValueError("Invalid column index for vertical line")
#
#         move_made = game.make_move(orientation, i, j)
#         if not move_made:
#             print("Illegal move or line already occupied. Try again.")
#
# game_state.make_move(move[0], move[1], move[2])
#
# # Print the outcome
# if game_state.outcome() == DotsBoxes.RED:
#     print("PUCT Player wins!")
# elif game_state.outcome() == DotsBoxes.BLUE:
#     print("Human Player wins!")
# else:
#     print("It's a draw!")