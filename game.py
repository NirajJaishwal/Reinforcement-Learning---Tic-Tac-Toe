import numpy as np

class Game:
    def __init__(self):
        self.board = [' '] * 9

    def play_game(self, player1, player2):

        self.reset_game()
        current_player, next_player = player1, player2

        while True:
            state = self.get_state()
            valid_actions = self.get_valid_actions(state)
            action = current_player.get_action(state, valid_actions)

            # If no action can be taken, it's a draw
            if action is None:
                return 'Draw'

            self.board[action] = current_player.side

            if self.check_winner(self.board, current_player.side):
                if current_player.side == 'O':
                    print()
                return current_player.side

            # Switch players
            current_player, next_player = next_player, current_player

    def reset_game(self):
        self.board = [' '] * 9

    def get_state(self):
        return ''.join(self.board)

    def get_valid_actions(self, state):
        board = np.array(list(state)).reshape((3, 3))
        return [i for i in range(9) if board.flatten()[i] == ' ']

    def check_winner(self, board, player):
        winning_combinations = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),
            (0, 3, 6), (1, 4, 7), (2, 5, 8),
            (0, 4, 8), (2, 4, 6)
        ]
        for combo in winning_combinations:
            if all(board[i] == player for i in combo):
                return True
        return False
