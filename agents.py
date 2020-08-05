import random

import torch

from alpha_net import AlphaNet
from mcts import mcts_search, Node
from utils.logger import *

test_log = 'test_logs/mcts.log'
logger = get_logger(__name__, test_log)

class Agent:
    def __init__(self, name='Name me please.'):
        self.name = name

    def get_next_move(self, state):
        "Return next game move. Must be legal action."
        raise NotImplementedError()

    def __repr__(self):
        return f'Agent name: {self.name}'


class RandomPlayer:
    def __init__(self, name='Random agent'):
        self.name = name

    def get_next_move(self, game):
        return random.choice(game.valid_actions)
    
    def __len__(self):
        return 1

    def __repr__(self):
        return '<Random agent>'

class HumanPlayer:
    def __init__(self, name='Human agent'):
        self.name = name

    def get_next_move(self, game):
        move = int(input('Please select a move: '))
        while move not in game.valid_actions:
            print("That's not a legal move. Try again.")
            move = int(input('Please select a move: '))
        return move

    def __len__(self):
        return 1
    
    def __repr__(self):
        return f'<{self.name}>'

class AlphaAgent:
    def __init__(self, n_simulations=100, C_puct=1.0, dirichlet_alpha=0.75, training=False, name='Alpha agent' ):
        self.net = AlphaNet()
        self.name = name
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.C_puct = C_puct
        self.training = training
        self.current_node = None
    
    def load_model(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = checkpoint['model_state_dict']
        self.net.load_state_dict(state_dict)

    def get_next_move(self, game):

        # logger.info(f'AlphaAgent Get Next Move \n {game}\n')

        if len(game.history) < 2:
            self.current_node = Node(game.state, player_id=game.player_turn, parent=None)
        else:
            opponent_move = game.history[-1]            
            self.current_node = self.current_node.edges[opponent_move]

            # logger.info(f'Updated current_node {self.current_node}')


        move = mcts_search(self.current_node, self.net, game, self.n_simulations, self.C_puct, self.dirichlet_alpha, self.training)
        self.current_node = self.current_node.edges[move]
        
        return move
    
    def __repr__(self):
        return '<AlphaNet agent>'
