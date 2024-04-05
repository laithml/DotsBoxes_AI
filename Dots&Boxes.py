class DotsBoxes:
    RED = 1
    BLUE = 2
    DRAW = 0
    ONGOING = -1

    def __init__(self):
        # Initialize horizontal and vertical lines
        self.horizontal_lines = [[False] * 7 for _ in range(8)]
        self.vertical_lines = [[False] * 8 for _ in range(7)]
        # Initialize box ownership (0: unclaimed, 1: RED, 2: BLUE)
        self.boxes = [[0] * 7 for _ in range(7)]
        self.score = [0, 0]  # RED, BLUE
        self.current_player = DotsBoxes.RED
        self.moves = 0
        self.history = []

    def legal_moves(self):
        # Returns a list of legal moves as tuples indicating the line position and orientation (h or v)
        moves = []
        for i in range(8):
            for j in range(7):
                if not self.horizontal_lines[i][j]:
                    moves.append(('h', i, j))
        for i in range(7):
            for j in range(8):
                if not self.vertical_lines[i][j]:
                    moves.append(('v', i, j))
        return moves

    def make_move(self, orientation, i, j):
        # Apply the move if legal and update the game state
        if orientation == 'h' and not self.horizontal_lines[i][j]:
            self.horizontal_lines[i][j] = True
        elif orientation == 'v' and not self.vertical_lines[i][j]:
            self.vertical_lines[i][j] = True
        else:
            return False  # Illegal move

        boxes_completed = self.update_boxes_after_move(orientation, i, j)
        if boxes_completed:
            self.score[self.current_player - 1] += boxes_completed
        else:
            self.current_player = self.other_player(self.current_player)
        self.moves += 1
        self.history.append((orientation, i, j))
        return True

    def update_boxes_after_move(self, orientation, i, j):
        completed_boxes = 0

        # Check surrounding boxes based on the orientation of the placed line
        if orientation == 'h':  # Horizontal line
            # Check the box above (if any)
            if i > 0:
                if self.horizontal_lines[i - 1][j] and self.vertical_lines[i - 1][j] and self.vertical_lines[i - 1][
                    j + 1]:
                    if self.boxes[i - 1][j] == 0:  # Box was unclaimed
                        self.boxes[i - 1][j] = self.current_player
                        completed_boxes += 1

            # Check the box below (if any)
            if i < 7:
                if self.horizontal_lines[i + 1][j] and self.vertical_lines[i][j] and self.vertical_lines[i][j + 1]:
                    if self.boxes[i][j] == 0:  # Box was unclaimed
                        self.boxes[i][j] = self.current_player
                        completed_boxes += 1

        elif orientation == 'v':  # Vertical line
            # Check the box to the left (if any)
            if j > 0:
                if self.vertical_lines[i][j - 1] and self.horizontal_lines[i][j - 1] and self.horizontal_lines[i + 1][
                    j - 1]:
                    if self.boxes[i][j - 1] == 0:  # Box was unclaimed
                        self.boxes[i][j - 1] = self.current_player
                        completed_boxes += 1

            # Check the box to the right (if any)
            if j < 7:
                if self.vertical_lines[i][j + 1] and self.horizontal_lines[i][j] and self.horizontal_lines[i + 1][j]:
                    if self.boxes[i][j] == 0:  # Box was unclaimed
                        self.boxes[i][j] = self.current_player
                        completed_boxes += 1

        return completed_boxes

    @staticmethod
    def other_player(player):
        return DotsBoxes.BLUE if player == DotsBoxes.RED else DotsBoxes.RED

    def clone(self):
        clone = DotsBoxes()
        clone.horizontal_lines = [row[:] for row in self.horizontal_lines]
        clone.vertical_lines = [row[:] for row in self.vertical_lines]
        clone.boxes = [row[:] for row in self.boxes]
        clone.score = self.score[:]
        clone.current_player = self.current_player
        clone.moves = self.moves
        clone.history = self.history[:]
        return clone

    def is_game_over(self):
        # Check if the game is over
        return self.moves == 49

    def outcome(self):
        # Determine the game's outcome
        if self.is_game_over():
            if self.score[0] > self.score[1]:
                return DotsBoxes.RED
            elif self.score[0] < self.score[1]:
                return DotsBoxes.BLUE
            else:
                return DotsBoxes.DRAW
        return DotsBoxes.ONGOING

    def print_board(self):
        # Column headers
        print("  " + "   ".join(str(i) for i in range(7)) + " ")
        print("  _" + ('_ _ ' * 7))

        for i in range(7):
            # Print horizontal lines and dots with row indices on the left
            horizontal_line = f"{i} ."
            for j in range(7):
                if self.horizontal_lines[i][j]:
                    horizontal_line += '---.'
                else:
                    horizontal_line += '   .'
            print(horizontal_line)

            # Print vertical lines and boxes, with row indices for these lines as well
            vertical_line = "  "
            for j in range(7):
                if self.vertical_lines[i][j]:
                    vertical_line += '|'
                else:
                    vertical_line += ' '
                # Check box ownership and display accordingly
                if self.boxes[i][j] == self.RED:
                    vertical_line += ' R '
                elif self.boxes[i][j] == self.BLUE:
                    vertical_line += ' B '
                else:
                    vertical_line += '   '
            vertical_line += '|'
            print(vertical_line)

        # Print the last row of horizontal lines with row index
        bottom_line = f"7 ."
        for j in range(7):
            if self.horizontal_lines[7][j]:
                bottom_line += '---.'
            else:
                bottom_line += '   .'
        print(bottom_line)
        print("  _" + ('_ _ ' * 7))


def play_game():
    game = DotsBoxes()
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
                orientation = input("Enter line orientation (h for horizontal(---), v for vertical(|)): ").lower()
                if orientation not in ['h', 'v']:
                    raise ValueError("Invalid orientation")
                i = int(input("Enter row index (0-7 for horizontal(---), 0-6 for vertical(|)): "))
                j = int(input("Enter column index (0-6 for horizontal(---), 0-7 for vertical(|)): "))
                if not (0 <= i < 8) or not (0 <= j < 8):
                    raise ValueError("Invalid indices")
                if orientation == 'h' and i == 7:
                    raise ValueError("Invalid row index for horizontal line")
                if orientation == 'v' and j == 7:
                    raise ValueError("Invalid column index for vertical line")

                move_made = game.make_move(orientation, i, j)
                if not move_made:
                    print("Illegal move or line already occupied. Try again.")
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

ss
if __name__ == "__main__":
    play_game()

# Test
print("Test 1:")
