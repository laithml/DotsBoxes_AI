from random import random

from DotsBoxes import DotsBoxes
from PUCTNode import PUCTNode


class PUCTPlayer:
    def __init__(self, game_state):
        self.root = PUCTNode(game_state)

    def choose_move(self, iterations):
        for _ in range(iterations):
            node = self.selection()
            outcome = self.simulation(node.game_state)
            self.backpropagation(node, outcome)

        return self.best_move()

    def selection(self):
        curr_node = self.root
        while not curr_node.is_fully_expanded() or curr_node.children:
            if not curr_node.is_fully_expanded():
                move = random.choice(curr_node.untried_moves)
                curr_node = curr_node.add_child(move)
                return curr_node
            else:
                curr_node = curr_node.choose_child()
        return curr_node

    def simulation(self, game_state):
        while game_state.outcome() == DotsBoxes.ONGOING:
            possible_moves = game_state.legal_moves()
            move = random.choice(possible_moves)
            game_state.make_move(move[0], move[1], move[2])
        return game_state.outcome()

    def backpropagation(self, node, outcome):
        curr_node = node
        while curr_node.has_parent():
            curr_node.visits += 1
            if outcome == curr_node.game_state.current_player:
                curr_node.wins += 1
            elif outcome == DotsBoxes.DRAW:
                curr_node.wins += 0.5
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