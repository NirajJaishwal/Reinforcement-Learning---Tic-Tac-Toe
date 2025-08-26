import numpy as np

class RandomPlayer:
    def __init__(self, side='X'):
        self.side = side

    def get_action(self, state, valid_actions):
        if not valid_actions:
            return None  # Return None if no valid actions are available
        return np.random.choice(valid_actions)

    def get_valid_actions(self, state):
        board = np.array(list(state)).reshape((3, 3))
        return [i for i in range(9) if board.flatten()[i] == ' ']
