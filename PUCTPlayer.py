import random

import torch

from MCTSNode import MCTSNode
from PolicyValueNetwork import PolicyValueNetwork


def decode_move(index):
    total_horizontal_moves = 8 * 7
    if index < total_horizontal_moves:
        orientation = 'h'
        row = index // 8
        column = index % 8
    else:
        orientation = 'v'
        index -= total_horizontal_moves
        row = index // (8 - 1)
        column = index % (8 - 1)

    return orientation, row, column


class PUCTPlayer:

    def __init__(self, game_state):
        self.root = MCTSNode(game_state)
        self.model = PolicyValueNetwork()

    def choose_move(self, iterations):
        for _ in range(iterations):
            self.selection_back_propagation()
        return self.best_move()

    def selection_back_propagation(self):
        curr_node = self.root

        while not curr_node.is_fully_expanded() or curr_node.children:
            if not curr_node.is_fully_expanded():
                move = random.choice(curr_node.untried_moves)
                curr_node = curr_node.add_child(move)

                # Get the encoded state and use the model to predict value and policy
                game_state_encoded = curr_node.game_state.encode_state()
                game_state_encoded = torch.tensor(game_state_encoded, dtype=torch.float32).unsqueeze(
                    0)  # Add batch dimension
                policy_output, value = self.model.forward(game_state_encoded)
                policy_output = policy_output.detach().numpy().flatten()  # Convert to numpy and flatten

                curr_node.Q = value.item()
                curr_node.N = 1

                # Decode all moves and assign probabilities
                decoded_moves = [(decode_move(i), policy_output[i]) for i in
                                 range(len(policy_output))]
                move_probs = {move: prob for move, prob in decoded_moves if move in curr_node.untried_moves}

                # Update untried moves with probabilities
                for move in curr_node.untried_moves:
                    tuple_move = tuple(move)
                    if tuple_move in move_probs:
                        move.probability = move_probs[tuple_move]

                break
            else:
                curr_node = curr_node.choose_child()

        # back propagation
        while curr_node.parent != None:
            par = curr_node.parent
            par.N += 1
            par.Q += (1 / par.N) * (par.Q - curr_node.Q)
            curr_node = curr_node.parent

    def best_move(self):
        best_N = -float("inf")
        best_move = None
        for child in self.root.children:
            if child.N > best_N:
                best_N = child.N
                best_move = child.move
        return best_move
