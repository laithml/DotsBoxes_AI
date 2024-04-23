import os

from DotsBoxes import DotsBoxes
from MCTSNode import MCTSNode
from PUCTPlayer import PUCTPlayer
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


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
        self.loss_fn = nn.MSELoss()  # For value loss
        self.policy_loss_fn = nn.CrossEntropyLoss()  # For policy loss

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

    def train_step(self, game_state, true_move, true_value):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.train()
        self.optimizer.zero_grad()

        # Convert game state to tensor
        game_state_tensor = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0).to(device)
        true_move_tensor = torch.tensor([true_move], dtype=torch.float32).to(device)
        true_value_tensor = torch.tensor([true_value], dtype=torch.float32).to(device)

        # Forward pass
        policy_logits, value = self.model(game_state_tensor)

        value_loss = self.loss_fn(value.squeeze(-1), true_value_tensor)
        policy_loss = self.policy_loss_fn(policy_logits, true_move_tensor)  # Direct use

        total_loss = value_loss + policy_loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item()

    def train_model(self):
        """ Train the model on the accumulated data. """
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.train()
        for game_state, move_index, target_value in self.data:
            self.train_step(game_state, move_index, target_value)

            # self.optimizer.zero_grad()
            # input_tensor = torch.tensor(game_state, dtype=torch.float32).unsqueeze(0).to(device)
            # policy_logits, value_output = self.model(input_tensor)

            # target_tensor_policy = torch.tensor([move_index], dtype=torch.float32).to(device)
            # loss_policy = F.cross_entropy(policy_logits, target_tensor_policy)

            # if value_output.numel() == 1:
            #     value_output_squeezed = value_output.view(1)
            # else:
            #     value_output_squeezed = value_output.squeeze()

            # target_tensor_value = torch.tensor([target_value], dtype=torch.float32).to(device)
            # loss_value = F.mse_loss(value_output_squeezed, target_tensor_value)

            # # Combine losses and perform backpropagation
            # loss = loss_policy + loss_value
            # loss.backward()
            # self.optimizer.step()

        # Clear data after training
        self.data = []

    def self_play(self):
        """ Simulate a game where the AI plays against itself and collect training data. """
        print("Starting a self-play session...")

        self.game.reset()
        self.ai = PUCTPlayer(self.game)
        puct_player1 = self.ai
        puct_player1.model = self.ai.model
        puct_player2 = PUCTPlayer(self.game)

        if os.path.exists("model.pth"):
            self.load_model("model.pth")

        while not self.game.is_game_over():
            # Current player
            if self.game.current_player == DotsBoxes.RED:
                puct_player = puct_player1
                # print("RED")
            else:
                puct_player = puct_player2
                # print("BLue")

            if self.game.current_player == DotsBoxes.RED:
                # Encode the move and the resulting game state
                game_state_encoded = self.game.encode_state()

            move = puct_player.choose_move(10)
            # AI chooses and makes a move
            valid_move, reward = self.game.make_move(move[0], move[1], move[2])

            if valid_move and self.game.history[-1][0] == DotsBoxes.BLUE:
                self.data[-1] = (self.data[-1][0], self.data[-1][1], self.data[-1][2] - reward)

            elif valid_move and self.game.history[-1][0] == DotsBoxes.RED:
                move_index = encode_move(move[0], move[1], move[2])
                policy_output = [0] * 112  # Total possible moves
                policy_output[move_index] = 1
                # Append to data set (state, policy, reward)
                self.data.append((game_state_encoded, policy_output, reward))

            # Update MCTS roots after the move
            puct_player1.root = MCTSNode(self.game)
            puct_player2.root = MCTSNode(self.game)

            # self.game.print_board()
            # Check game outcome
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

        self.data = [(gs, policy, reward) for (gs, policy, _) in self.data]

        # Final scores
        print(f"Final Scores - RED: {self.game.score[0]}, BLUE: {self.game.score[1]}")

    def play_game(self):
        self.game.reset()
        self.ai = PUCTPlayer(self.game)
        puct_player = self.ai
        puct_player.model = self.ai.model

        if os.path.exists("model.pth"):
            self.load_model("model.pth")

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
                    move = puct_player.choose_move(2000)
                    print(f"MCTSPlayer chooses column {move}")
                    move_made, _ = self.game.make_move(move[0], move[1], move[2])
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
                        move_made, _ = self.game.make_move(orientation, i, j)
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
