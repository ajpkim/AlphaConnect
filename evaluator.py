import logging
import random
import numpy as np

from game.play_game import play_game
from game.connect4 import Connect4


class Evaluator:
	def __init__(self, game):
		"""
		Class to evaluate 2 agents in given game
		
		Args: 
			- game: game class, not an instance. 
		"""
		self.game = game

	def evaluate(self, agent_1, agent_2, n_episodes):
		"Method for comparing 2 agents. Returns dict of totals and dict of percentages."
		
		results, percentages = {}, {}
		outcomes = np.array([play_game(self.game(), agent_1, agent_2) for n in range(n_episodes)])
	
		results[agent_1.name] = (outcomes == agent_1.name).sum()
		results[agent_2.name] = (outcomes == agent_2.name).sum()
		results['tie'] = len(outcomes) - results[agent_1.name] - results[agent_2.name]

		percentages[agent_1.name] = round(results[agent_1.name] / n_episodes, 4)
		percentages[agent_2.name] = round(results[agent_2.name] / n_episodes, 4)
		percentages['tie'] = round(results['tie'] / n_episodes, 4)
		
		return results, percentages

	def __repr__(self):
		return f'<Evaluator for game {self.game}>'
		
		
		


		

