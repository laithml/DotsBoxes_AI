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

                #TODO: put the situation inside the network to get the value and p
                value,p = PolicyValueNetwork.forward(curr_node.game_state)

                # p = np.zeros(len(curr_node.game_state.legal_moves())) #we should have it from the network

                curr_node.Q = value
                curr_node.N = 1
                possible_moves = curr_node.game_state.legal_moves()

                for i in range(len(possible_moves)):
                    possible_moves[i].P = p[i]
                break
            else:
                curr_node = curr_node.choose_child_puct()

        # back propagation
        while curr_node != None:
            curr_node.visits += 1
            curr_player = curr_node.game_state.current_player
            if value == curr_player:
                curr_node.wins += value
            elif value == curr_player.other_player(curr_player):
                curr_node.wins += (1 - value)
            curr_node = curr_node.parent

    def best_move(self):
        best_win_rate = -float("inf")
        best_move = None
        for child in self.root.children:
            win_rate = child.wins / child.visits
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_move = child.move
        return best_move
