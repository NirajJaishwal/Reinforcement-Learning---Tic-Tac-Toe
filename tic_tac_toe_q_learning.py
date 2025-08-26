import numpy as np
import pickle

class TicTacToeQLearning:
    def __init__(self, side='X', learning_rate=0.01, discount_factor=0.9, exploration_rate=0.1, exploration_decay=0.9, first_player = False):
        self.q_table = {}  # Dictionary to store Q-values
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.side = side  # The side this player plays as ('X' or 'O')
        self.exploration_decay = exploration_decay
        self.first_player = first_player
        self.reset_metrics()
        self.episode_results = []

    def reset_metrics(self):
        self.win_count = 0
        self.lose_count = 0
        self.draw_count = 0
        

    def get_action(self, state, valid_actions):

        if not valid_actions:
            return None

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(valid_actions)

        q_values = [self.q_table.get((state, action), 0) for action in valid_actions]
        max_q_value = max(q_values)
        max_q_indices = [i for i, q_value in enumerate(q_values) if q_value == max_q_value]

        best_action_index = np.random.choice(max_q_indices)
        return valid_actions[best_action_index]

    def update_q_table(self, state, action, reward, next_state):

        current_q_value = self.q_table.get((state, action), 0)

        next_valid_actions = self.get_valid_actions(next_state)
        max_next_q_value = max([self.q_table.get((next_state, a), 0) for a in next_valid_actions], default=0)
        #print(max_next_q_value)
        self.q_table[(state, action)] = (1 - self.learning_rate) * current_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value)

    def get_valid_actions(self, state):

        board = np.array(list(state)).reshape((3, 3))
        return [i for i in range(9) if board.flatten()[i] == ' ']

    def train(self, episodes):

        for episode in range(episodes):
            self.reset_game()
            opponent = 'O' if self.side == 'X' else 'X'
            if self.first_player:
                current_player = self.side
            else:
                current_player = opponent

            while True:

                if current_player == self.side:
                    state = self.get_state()
                    valid_actions = self.get_valid_actions(state)
                    action = self.get_action(state, valid_actions)

                    # If no action can be made, the game is a draw
                    if action is None:
                        reward = 0
                        self.update_q_table(state, action, reward, self.get_state())
                        self.draw_count += 1
                        break

                    self.board[action] = current_player

                    # Check for a win or lose
                    if self.check_winner(self.board, current_player):
                        reward = 1 if current_player == self.side else -1
                        self.update_q_table(state, action, reward, self.get_state())
                        if current_player == self.side:
                            self.win_count += 1
                        else:
                            self.lose_count += 1
                        break

                    
                else:
                    valid_actions = self.get_valid_actions(self.get_state())
                    if valid_actions:
                        self.board[np.random.choice(valid_actions)] = current_player
                        if self.check_winner(self.board, current_player):
                            reward = 1 if current_player == self.side else -1
                            self.update_q_table(state, action, reward, self.get_state())
                            if current_player != self.side:
                                self.lose_count += 1
                            else:
                                self.win_count += 1
                            break
                            break
                        if len(valid_actions) != 9:
                            next_state = self.get_state()
                            self.update_q_table(state, action, 0, next_state)
                    else:
                        reward = 0
                        self.update_q_table(state, action, reward, self.get_state())
                        self.draw_count += 1
                        break
                     
                # Alternate between players
                current_player = opponent if current_player == self.side else self.side

            # Decay exploration rate
            self.exploration_rate *= self.exploration_decay

            if (episode + 1) % 100 == 0:
                self.episode_results.append((episode + 1, self.win_count, self.lose_count, self.draw_count))
                self.reset_metrics()

    def reset_game(self):
        self.board = [' '] * 9

    def get_state(self):

        return ''.join(self.board)

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
    
    def save_q_table(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)
    
    def load_q_table(self, filename):
        with open(filename, 'rb') as f:
            self.q_table = pickle.load(f)

