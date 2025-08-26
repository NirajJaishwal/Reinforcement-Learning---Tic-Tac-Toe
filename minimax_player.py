import numpy as np

class MinimaxPlayer:
    def __init__(self, side='X'):
        self.side = side

    def get_action(self, state, valid_actions):
        board = np.array(list(state)).reshape((3, 3))
        best_score = float('-inf')
        best_action = None

        for action in valid_actions:
            row, col = divmod(action, 3)
            board[row, col] = self.side
            score = self.minimax(board, False)
            board[row, col] = ' '

            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def minimax(self, board, is_maximizing):
        opponent = 'O' if self.side == 'X' else 'X'

        if self.check_winner(board, self.side):
            return 10
        elif self.check_winner(board, opponent):
            return -10
        elif np.all(board != ' '):
            return 0

        if is_maximizing:
            best_score = float('-inf')
            for action in range(9):
                row, col = divmod(action, 3)
                if board[row, col] == ' ':
                    board[row, col] = self.side
                    score = self.minimax(board, False)
                    board[row, col] = ' '
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for action in range(9):
                row, col = divmod(action, 3)
                if board[row, col] == ' ':
                    board[row, col] = opponent
                    score = self.minimax(board, True)
                    board[row, col] = ' '
                    best_score = min(score, best_score)
            return best_score

    def check_winner(self, board, player):
        winning_combinations = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)],
            [(0, 0), (1, 0), (2, 0)],
            [(0, 1), (1, 1), (2, 1)],
            [(0, 2), (1, 2), (2, 2)],
            [(0, 0), (1, 1), (2, 2)],
            [(2, 0), (1, 1), (0, 2)]
        ]

        # Ensure board is a numpy array
        board = np.array(board).reshape((3, 3))

        for combination in winning_combinations:
            if all(board[row, col] == player for row, col in combination):
                return True
        return False

    def get_valid_actions(self, state):
        board = np.array(list(state)).reshape((3, 3))
        return [i for i in range(9) if board.flatten()[i] == ' ']
