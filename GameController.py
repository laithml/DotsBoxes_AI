import os

from DotsBoxes import DotsBoxes
from MCTSNode import MCTSNode
from PUCTPlayer import PUCTPlayer
import torch
import torch.optim as optim
import torch.nn.functional as F


def encode_move(orientation, row, column):
    if orientation == 'h':
        # Calculate index for horizontal moves
        return row * 7 + column
    elif orientation == 'v':
        # Calculate index for vertical moves, starting at 56
        return 56 + row * 8 + column
    else:
        raise ValueError("Invalid move orientation")


class GameController:
    def __init__(self):
        self.game = DotsBoxes()
        self.ai = PUCTPlayer(self.game)
        self.model = self.ai.model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Example optimizer
        self.data = []  # Store training data

    def train(self, iterations, epochs):
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for _ in range(iterations):
                self.self_play()
            print("Training on gathered data...")

            self.train_model()
            self.model.save("model.pth", self.optimizer)

            self.game.reset()

        print("Training complete.")

    def load_model(self, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('using', device)
        optimizer_state_dict = self.model.load(path, device)
        self.optimizer.load_state_dict(optimizer_state_dict)
        print("Optimizer state has been loaded.")

    def train_model(self):
        """ Train the model on the accumulated data. """
        self.model.train()
        for game_state, target_policy, target_value in self.data:
            self.optimizer.zero_grad()
            input_tensor = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0)
            if torch.cuda.is_available():
                input_tensor = input_tensor.cuda()  # Move tensor to GPU
            policy_output, value_output = self.model(input_tensor)
            target_policy = [target_policy]
            print(policy_output)
            print(torch.tensor(target_policy).long().cuda())
            loss_policy = F.cross_entropy(policy_output, torch.tensor(target_policy).float().cuda())
            loss_value = F.mse_loss(value_output.squeeze(), torch.tensor([target_value], dtype=torch.float32).cuda())

            # combined the loss and add the regularization term for the wightes
            l2_reg = torch.tensor(0., requires_grad=True)
            for param in self.model.parameters():
                l2_reg += torch.norm(param)
            loss = loss_policy + loss_value + 0.01 * l2_reg
            loss.backward()
            self.optimizer.step()

        self.data = []

    def self_play(self):
        """ Simulate a game where the AI plays against itself and collect training data. """
        print("Starting a self-play session...")
        if os.path.exists("model.pth"):
            self.load_model("model.pth")

        self.game.reset()
        puct_player1 = self.ai
        puct_player2 = PUCTPlayer(self.game)

        while not self.game.is_game_over():
            # Current player
            if self.game.current_player == DotsBoxes.RED:
                puct_player = puct_player1
                print("RED")
            else:
                puct_player = puct_player2
                print("BLue")

            # AI chooses and makes a move
            move = puct_player.choose_move(10)
            valid_move = self.game.make_move(move[0], move[1], move[2])

            if valid_move and puct_player == puct_player1:
                # Encode the move and the resulting game state
                game_state_encoded = self.game.encode_state()
                move_index = encode_move(move[0], move[1], move[2])
                policy_output = [0] * 112  # Total possible moves
                policy_output[move_index] = 1
                # Append to data set (state, policy, reward)
                self.data.append((game_state_encoded, policy_output, 0))

            # Update MCTS roots after the move
            puct_player.root = MCTSNode(self.game)

            # Check game outcome
            outcome = self.game.outcome()
            if outcome != DotsBoxes.ONGOING:
                break

        # Game over, declare winner
        if outcome == DotsBoxes.RED:
            print("Game over! RED wins!")
            for i in range(len(self.data)):
                new_tuple = (self.data[i][0], self.data[i][1], 1)
                self.data[i] = new_tuple
        elif outcome == DotsBoxes.BLUE:
            print("Game over! BLUE wins!")
            for i in range(len(self.data)):
                new_tuple = (self.data[i][0], self.data[i][1], 0)
                self.data[i] = new_tuple
        else:
            print("Game over! It's a draw!")
            for i in range(len(self.data)):
                new_tuple = (self.data[i][0], self.data[i][1], 0.5)
                self.data[i] = new_tuple
        # Final scores
        print(f"Final Scores - RED: {self.game.score[0]}, BLUE: {self.game.score[1]}")

    def play_game(self):
        if os.path.exists("model.pth"):
            self.load_model("model.pth")
        self.game.reset()
        puct_player = self.ai
        print("Welcome to Dots and Boxes!")
        self.game.print_board()
        while not self.game.is_game_over():
            # Print current scores
            print(f"Scores - RED: {self.game.score[0]}, BLUE: {self.game.score[1]}")

            # Current player
            player_color = "RED" if self.game.current_player == DotsBoxes.RED else "BLUE"
            print(f"{player_color}'s turn")

            # Get move from current player
            move_made = False
            while not move_made:
                if self.game.current_player == DotsBoxes.RED:
                    # MCTSPlayer's turn
                    print("MCTSPlayer (RED) is thinking...")
                    move = puct_player.choose_move(1000)
                    print(f"MCTSPlayer chooses column {move}")
                    move_made = self.game.make_move(move[0], move[1], move[2])
                    print("MCTSPlayer end")
                else:
                    try:
                        orientation = input(
                            "Enter line orientation (h for horizontal(---), v for vertical(|)): ").lower()
                        if orientation not in ['h', 'v']:
                            raise ValueError("Invalid orientation")
                        if (orientation == 'h'):
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
                        move_made = self.game.make_move(orientation, i, j)
                        if not move_made:
                            print("Illegal move or line already occupied. Try again.")
                    except ValueError as e:
                        print(f"Error: {e}. Please try again.")

                if self.game.current_player == DotsBoxes.RED:
                    puct_player.root = MCTSNode(self.game)

            self.game.print_board()
            # Check for game outcome
            outcome = self.game.outcome()
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
        print(f"Final Scores - RED: {self.game.score[0]}, BLUE: {self.game.score[1]}")

    def menu(self):
        """ Display menu options to the user. """
        option = 0
        while option != 9:
            print("\n1. Train the AI")
            print("2. Play against the AI")
            print("9. Exit")
            try:
                option = int(input("Select an option: "))
            except ValueError:
                print("Please enter a valid number.")
            if option == 1:
                try:
                    iterations = int(input("Enter number of iterations per epoch: "))
                    epochs = int(input("Enter number of epochs: "))
                except ValueError:
                    print("Please enter a valid number.")
                self.train(iterations, epochs)
            elif option == 2:
                self.play_game()
            elif option == 9:
                print("Exiting...")
            else:
                print("Invalid option, please choose again.")


if __name__ == "__main__":
    controller = GameController()
    controller.menu()
