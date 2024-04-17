import math

from DotsBoxes import DotsBoxes

class MCTSNode:
    def __init__(self, board, parent=None,move = [None, None, None]):
        self.wins = 0
        self.visits = 0
        if parent is None:
            self.visits = 1
        self.parent = parent
        self.move = move
        self.game_state = board.clone()
        self.children = []
        self.untried_moves = DotsBoxes.legal_moves(self.game_state)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def add_child(self, move):
        curr_state = self.game_state.clone() # no need maybe
        curr_state.make_move(move[0], move[1], move[2])

        if move in self.untried_moves:
            self.untried_moves.remove(move)

        child = MCTSNode(curr_state, self, move)
        self.children.append(child)
        return child

    def has_parent(self):
        return self.parent is not None

    def choose_child(self, c_param=math.sqrt(2)):
        if len(self.children) == 0:
            return None

        log_visits = math.log(self.visits)
        best_value = -float("inf")
        best_child = None

        for child in self.children:
            win_avg = child.wins / child.visits
            exploration_term = c_param * math.sqrt(log_visits / child.visits)

            uct = win_avg + exploration_term # UCT there's diff

            if uct > best_value:
                best_child = child
                best_value = uct

        return best_child
