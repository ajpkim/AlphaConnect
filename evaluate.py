import numpy as np

from game.play_game import play_game

def evaluate(Game, agent1, agent2, n_episodes):
	"""
	Evaluate 2 agents in given game across n_episodes.

	Args:
	     - Game: Game class to evaluate agents on
	     - agents: agents which both implement get_next_move()
	     - n_episodes: number of games to play with each agent as first player
	        	      (total games played is n_episodes * 2)

	Returns:
		- results: Nested disctionaries for when each agent plays first.
                           { {agent1: wins, agent2: wins, tie: ties},  # agent1 goes first
                             {agent1: wins, agent2: wins, tie: ties} }  # agent2 goes first
        """
	results = {}

	agent1_as_player1_results = {}
	outcomes = np.array([play_game(Game(), agent1, agent2, shuffle_order=False) for n in range(n_episodes)])
	agent1_as_player1_results[agent1.name] = (outcomes == agent1.name).sum()
	agent1_as_player1_results[agent2.name] = (outcomes == agent2.name).sum()
	agent1_as_player1_results['tie'] = (outcomes == 'tie').sum()
	results[f"{agent1.name} as player1"] = agent1_as_player1_results

	agent2_as_player1_results = {}
	outcomes = np.array([play_game(Game(), agent2, agent1, shuffle_order=False) for n in range(n_episodes)])
	agent2_as_player1_results[agent1.name] = (outcomes == agent1.name).sum()
	agent2_as_player1_results[agent2.name] = (outcomes == agent2.name).sum()
	agent2_as_player1_results['tie'] = (outcomes == 'tie').sum()
	results[f"{agent2.name} as player1"] = agent2_as_player1_results

	return results
