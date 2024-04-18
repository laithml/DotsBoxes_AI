from random import random

from MCTSNode import MCTSNode


class PUCTPlayer:
    # TODO: decoding the policy from the NN, to move, then make the move,
    #  there's no need to simulate and no need to choose the best move, because we know the best from the NN

    def __init__(self, game_state):
        self.root = MCTSNode(game_state)

    def choose_move(self, iterations):
        for _ in range(iterations):
            node = self.selection()
            outcome = self.simulation(node.game_state)
            self.backpropagation(node, outcome)  # TODO: GET THE VALUE

        return self.best_move()

    def selection(self):
        curr_node = self.root
        while not curr_node.is_fully_expanded() or curr_node.children:
            if not curr_node.is_fully_expanded():
                move = random.choice(curr_node.untried_moves)
                curr_node = curr_node.add_child(move)
                return curr_node
            else:
                curr_node = curr_node.choose_child_puct()
        return curr_node

    def backpropagation(self, node, outcome, value):
        curr_node = node
        while curr_node.has_parent():
            curr_node.visits += 1
            curr_player = curr_node.game_state.current_player
            if outcome == curr_player:
                curr_node.wins += value
            elif outcome == curr_player.other_player(curr_player):
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
