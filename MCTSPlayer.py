import random

from DotsBoxes import DotsBoxes
from MCTSNode import MCTSNode


class MCTSPlayer:

    def __init__(self, game_state):
        self.root = MCTSNode(game_state)

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
                # If the node has untried moves, select one and expand the tree by adding a new child node.
                move = random.choice(curr_node.untried_moves)  # Randomly select an untried move
                curr_node = curr_node.add_child(move)  # Expand the tree
                return curr_node  # Return the newly expanded node for simulation
            else:
                # If the node is fully expanded, use the UCT method to choose the next child node to explore.
                curr_node = curr_node.choose_child()
        return curr_node

    def simulation(self, game_state):
        while game_state.outcome() == DotsBoxes.ONGOING:
            possible_moves = game_state.legal_moves()
            move = random.choice(possible_moves)
            game_state.make_move(move[0],move[1],move[2])
            # if there's a win, for the opponent, play this move and don't let the opponent win
        return game_state.outcome()

    def backpropagation(self, node, outcome):
        curr_node = node
        while curr_node.has_parent():
            curr_node.visits += 1
            if outcome == curr_node.game_state.current_player:
                curr_node.wins += 1
            curr_node = curr_node.parent

        # DRAW, LOSE, WIN


    def best_move(self):
        best_win_rate = -float("inf")
        best_move = None
        for child in self.root.children:
            win_rate = child.wins / child.visits
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_move = child.move
        return best_move


def main():
    print("hi!")
    game = DotsBoxes()
    mcts_player = MCTSPlayer(game)
    print("Welcome to Dots and Boxes!")
    game.print_board()

    while not game.is_game_over():
        # Print current scores
        print(f"Scores - RED: {game.score[DotsBoxes.RED - 1]}, BLUE: {game.score[DotsBoxes.BLUE - 1]}")


        # Current player
        player_color = "RED" if game.current_player == DotsBoxes.RED else "BLUE"
        print(f"{player_color}'s turn")


        # Get move from current player
        move_made = False
        while not move_made:
            try:

                if game.current_player == DotsBoxes.RED:
                    # MCTSPlayer's turn
                    print("MCTSPlayer (RED) is thinking...")
                    move = mcts_player.choose_move(10)
                    print(f"MCTSPlayer chooses column {move}")
                    move_made = game.make_move(move[0], move[1], move[2])
                    print("MCTSPlayer end")

                else:
                    orientation = input("Enter line orientation (h for horizontal(---), v for vertical(|)): ").lower()
                    if orientation not in ['h', 'v']:
                        raise ValueError("Invalid orientation")
                    if(orientation == 'h'):
                        i = int(input("Enter row index (0-7): "))
                        j = int(input("Enter column index (0-6): "))
                    else:
                        i = int(input("Enter row index (0-6): "))
                        j = int(input("Enter column index (0-7): "))
                    if not (0 <= i < 8) or not (0 <= j < 8):
                        raise ValueError("Invalid indices")
                    if orientation == 'h' and j == 7:
                        raise ValueError("Invalid row index for horizontal line")
                    if orientation == 'v' and i == 7:
                        raise ValueError("Invalid column index for vertical line")

                    move_made = game.make_move(orientation, i, j)
                    if not move_made:
                        print("Illegal move or line already occupied. Try again.")

                if game.current_player == DotsBoxes.RED:
                    mcts_player.root = MCTSNode(game)
            except ValueError as e:
                print(f"Error: {e}. Please try again.")

        game.print_board()

        # Check for game outcome
        outcome = game.outcome()
        if outcome != DotsBoxes.ONGOING:
            break

    # Game over, declare winner
    if outcome == DotsBoxes.RED:
        print("Game over! RED wins!")
    elif outcome == DotsBoxes.BLUE:
        print("Game over! BLUE wins!")
    else:
        print("Game over! It's a draw!")

    # Final scores
    print(f"Final Scores - RED: {game.score[DotsBoxes.RED - 1]}, BLUE: {game.score[DotsBoxes.BLUE - 1]}")
#
# if __name__ == "__main__":
#     play_game()



main()