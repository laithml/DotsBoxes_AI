import random

import torch

from DotsBoxes import DotsBoxes
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
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def choose_move(self, iterations):
        for _ in range(iterations):
            self.selection_back_propagation()
            # self.print_tree()
        return self.best_move()

    def selection_back_propagation(self):
        curr_node = self.root
        while not curr_node.is_fully_expanded() or curr_node.children:
            if not curr_node.is_fully_expanded():

                curr_node.untried_moves = DotsBoxes.legal_moves(curr_node.game_state)

                move = random.choice(curr_node.untried_moves)  # Select a move from untried moves
                    # print(f"Trying move: {move} from state:\n{curr_node.game_state}")
                curr_node.untried_moves.remove(move)  # Remove the selected move from untried moves
                curr_node = curr_node.add_child(move)  # Expand this move into a new child node

                # Get the encoded state and use the model to predict value and policy
                game_state_encoded = curr_node.game_state.encode_state()
                game_state_encoded = torch.tensor(game_state_encoded, dtype=torch.float32).unsqueeze(0)
                if torch.cuda.is_available():
                    game_state_encoded = game_state_encoded.cuda()
                policy_output, value = self.model.forward(game_state_encoded)
                policy_output = policy_output.detach().cpu().numpy().flatten()

                curr_node.Q = value.item()
                curr_node.N = 1

                # Decode all moves and assign probabilities
                decoded_moves = [(decode_move(i), policy_output[i]) for i in range(len(policy_output))]
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
        while curr_node.parent is not None:
            par = curr_node.parent
            par.N += 1
            par.Q += (1 / par.N) * (par.Q - curr_node.Q)
            curr_node = curr_node.parent

    def best_move(self):
        if not self.root.children:
            print("No moves explored.")
            return None

        best_N = -float("inf")
        best_move = None
        for child in self.root.children:
            if child.N > best_N:
                best_N = child.N
                best_move = child.move

        if best_move is not None:
            print("Best move:", best_move)
        else:
            print("No valid move found.")
        return best_move

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
            print("Root node (current player is {}):".format(
                "RED" if node.game_state.current_player == DotsBoxes.RED else "BLUE"))

        for child in node.children:
            move_str = "Move: {}, N: {}, Q: {:.2f}".format(child.move, child.N, child.Q)
            print(indent + move_str)
            if child.children:
                self.print_tree(child, indent + "    ")
