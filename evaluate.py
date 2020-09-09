import numpy as np

from game.play_game import play_game

def evaluate(Game, agent_1, agent_2, n_episodes):
	"""
	Function for comparing 2 agents. 
	
	Args:
		- Game: Game class to evaluate agents on
		- agents: agents which implement get_next_move()
		- n_episodes: number of games to play with each agent as first player
			(total games played is n_episodes * 2)

	Returns:
		results : {k=player1 agent, v=results}"""

	results = {}

	a1_as_p1_results = {}
	outcomes = np.array([play_game(Game(), agent_1, agent_2, shuffle_order=False) for n in range(n_episodes)])
	a1_as_p1_results[agent_1.name] = (outcomes == agent_1.name).sum()
	a1_as_p1_results[agent_2.name] = (outcomes == agent_2.name).sum()
	a1_as_p1_results['tie'] = (outcomes == 'tie').sum()
	results[f"{agent_1.name} as p1"] = a1_as_p1_results

	a2_as_p1_results = {}
	outcomes = np.array([play_game(Game(), agent_2, agent_1, shuffle_order=False) for n in range(n_episodes)])
	a2_as_p1_results[agent_1.name] = (outcomes == agent_1.name).sum()
	a2_as_p1_results[agent_2.name] = (outcomes == agent_2.name).sum()
	a2_as_p1_results['tie'] = (outcomes == 'tie').sum()
	results[f"{agent_2.name} as p1"] = a2_as_p1_results

	return results
