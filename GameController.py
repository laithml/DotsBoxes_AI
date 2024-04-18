from DotsBoxes import DotsBoxes
from PUCTPlayer import PUCTPlayer


class GameController:
    def __init__(self):
        self.game = DotsBoxes()
        self.ai = PUCTPlayer(self.game)
        self.model = self.ai.model

    def train(self, iterations, epochs):
        """ Train the model using self-play for a given number of iterations and epochs. """
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for _ in range(iterations):
                self.self_play()
            print("Training on gathered data...")
            # Train the model on gathered data
            self.model.train()
            print("Training complete.")

    def self_play(self):
        """ Simulate a game where the AI plays against itself. """
        print("Starting a self-play session...")
        self.game.reset()
        while not self.game.is_game_over():
            move = self.ai.choose_move(100)
            self.game.make_move(move[0], move[1], move[2])

    def play_game(self):
        """ Play a game against the AI. """
        print("Welcome to Dots and Boxes!")
        self.game.print_board()

        while not self.game.is_game_over():
            # Print current scores
            print(f"Scores - RED: {self.game.score[self.game.RED - 1]}, BLUE: {self.game.score[self.game.BLUE - 1]}")

            # Current player
            player_color = "RED" if self.game.current_player == self.game.RED else "BLUE"
            print(f"{player_color}'s turn")

            # Get move from current player
            move_made = False
            while not move_made:
                try:
                    orientation_input = input(
                        "Enter line orientation (h for horizontal(---), v for vertical(|)): ").lower()
                    orientation = 0 if orientation_input == 'h' else 1  # Assume 0 is horizontal and 1 is vertical for the move tuple
                    if orientation_input not in ['h', 'v']:
                        raise ValueError("Invalid orientation")

                    if orientation_input == 'h':
                        i = int(input("Enter row index (0-7): "))
                        j = int(input("Enter column index (0-6): "))
                    else:
                        i = int(input("Enter row index (0-6): "))
                        j = int(input("Enter column index (0-7): "))

                    if not (0 <= i < 8) or not (0 <= j < 8):
                        raise ValueError("Invalid indices")
                    if orientation_input == 'h' and j == 7:
                        raise ValueError("Invalid column index for horizontal line")
                    if orientation_input == 'v' and i == 7:
                        raise ValueError("Invalid row index for vertical line")

                    move_made = self.game.make_move(orientation, i, j)
                    if not move_made:
                        print("Illegal move or line already occupied. Try again.")
                except ValueError as e:
                    print(f"Error: {e}. Please try again.")

            self.game.print_board()

            # Check for game outcome
            outcome = self.game.outcome()
            if outcome != self.game.ONGOING:
                break

        # Game over, declare winner
        if outcome == self.game.RED:
            print("Game over! RED wins!")
        elif outcome == self.game.BLUE:
            print("Game over! BLUE wins!")
        else:
            print("Game over! It's a draw!")

        # Final scores
        print(f"Final Scores - RED: {self.game.score[self.game.RED - 1]}, BLUE: {self.game.score[self.game.BLUE - 1]}")

    def menu(self):
        """ Display menu options to the user. """
        option = 0
        while option not in [1, 2, 9]:
            print("\n1. Train the AI")
            print("2. Play against the AI")
            print("9. Exit")
            try:
                option = int(input("Select an option: "))
                if option == 1:
                    iterations = int(input("Enter number of iterations per epoch: "))
                    epochs = int(input("Enter number of epochs: "))
                    self.train(iterations, epochs)
                elif option == 2:
                    self.play_game()
                elif option == 9:
                    print("Exiting...")
                else:
                    print("Invalid option, please choose again.")
            except ValueError:
                print("Please enter a valid number.")


# Example usage:
if __name__ == "__main__":
    controller = GameController()
    controller.menu()
