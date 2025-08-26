from tic_tac_toe_q_learning import TicTacToeQLearning
from minimax_player import MinimaxPlayer
from random_player import RandomPlayer
from game import Game
import numpy as np
import matplotlib.pyplot as plt

def train_agent(side='X'):
    agent = TicTacToeQLearning(side=side,first_player = False)
    agent.train(20000)
    plot_training_results(agent.episode_results)
    agent.save_q_table('q_table_P2_X.pkl')


def test_agent_vs_random():
    # Initialize the Q-learning agent and the random player
    agent = TicTacToeQLearning(side='X', exploration_rate=0)
    agent.load_q_table('q_table_P1_X.pkl')  
    random_player = RandomPlayer(side='O')
    plot(agent,random_player)


def test_agent_vs_minimax():
    # Initialize the Q-learning agent and the Minimax player
    agent = TicTacToeQLearning(side='X', exploration_rate=0)
    agent.load_q_table('q_table_P1_X.pkl')  
    minimax_player = MinimaxPlayer(side='O')
    plot(agent,minimax_player)

def test_minimax_vs_random():
    
    random_player = RandomPlayer(side='X')
    minimax_player = MinimaxPlayer(side='O')
    plot(random_player,minimax_player)
    
def test_agent_vs_agent():
    agent1 = TicTacToeQLearning(side='X',exploration_rate=0)
    agent1.load_q_table('q_table_P1_X.pkl')
    agent2 = TicTacToeQLearning(side='O',exploration_rate=0)
    #train_agent(side='O')
    agent2.load_q_table('q_table_P2_O.pkl')
    plot(agent1,agent2)

def plot_training_results(results):
    episodes, wins, losses, draws = zip(*results)
    plt.figure(figsize=(12, 6))

    # Define the colors for each stack
    colors = ['g', 'r', 'b']  # Green for Wins, Red for Losses, Blue for Draws

    # Plot the stackplot with the colors
    plt.stackplot(episodes, wins, losses, draws, labels=['Wins', 'Losses', 'Draws'], colors=colors, alpha=0.6)

    plt.xlabel('Episodes')
    plt.ylabel('Count')
    plt.title('Training Performance of Tic-Tac-Toe Q-Learning Agent')
    plt.legend(loc='upper left')
    plt.savefig('training_performance.png')
    plt.show()


def plot(player_1,player_2): 
    game = Game()  
    num_trials = 10  
    num_games_per_trial = 100  

    all_results = []  

    for trial in range(num_trials):
        results = {'X': 0, 'O': 0, 'Draw': 0}
        for _ in range(num_games_per_trial):
            result = game.play_game(player_1, player_2)
            results[result] += 1

        all_results.append(results)

    x_labels = [f"Trial {i+1}" for i in range(num_trials)]
    x = np.arange(len(x_labels))
    wins = [res['X'] for res in all_results]
    losses = [res['O'] for res in all_results]
    draws = [res['Draw'] for res in all_results]

    plt.figure(figsize=(10, 6))
    plt.bar(x, wins, label='Wins', color='g')
    plt.bar(x, losses, bottom=wins, label='Losses', color='r')
    plt.bar(x, draws, bottom=np.array(wins) + np.array(losses), label='Draws', color='b')

    plt.xlabel('Trial Number')
    plt.ylabel('Number of Games')
    plt.title('Results of 100 Games vs Minimax Player (10 Trials)')
    plt.xticks(x, x_labels, rotation=45)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig('results_stacked_bar_chart_vs_minimax.png')
    plt.show()

    avg_wins = np.mean(wins)
    avg_losses = np.mean(losses)
    avg_draws = np.mean(draws)

    print("Average Results after 10 Trials:")
    print(f"Average Wins: {avg_wins:.2f}")
    print(f"Average Losses: {avg_losses:.2f}")
    print(f"Average Draws: {avg_draws:.2f}")

if __name__ == '__main__':
    # train_agent()
    #test_agent_vs_random()
    #test_agent_vs_minimax()
    #test_minimax_vs_random()
    # test_agent_vs_agent()
