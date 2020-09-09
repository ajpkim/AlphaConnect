import copy
import random

import numpy as np
import torch

from alpha_net import AlphaNet
from mcts import mcts_search, Node, backup, select_action, select_leaf
from utils.logger import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


class Row0Player:
    def __init__(self, name='Row 0 Player'):
        self.name = name    

    def get_next_move(self, game):
        move = 0
        while move not in game.valid_actions:
            move += 1
        return move

    def __len__(self):
        return 1
    
    def __repr__(self):
        return f'<{self.name}>'


class AlphaAgent:
    def __init__(self, n_simulations=100, C_puct=2.0, dirichlet_alpha=0.75, training=False, name='Alpha agent' ):
        self.net = AlphaNet().to(device)
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

        if len(game.history) < 2:
            self.current_node = Node(game.state, player_id=game.player_turn, parent=None)
        else:
            opponent_move = game.history[-1]            
            self.current_node = self.current_node.edges[opponent_move]

        move = mcts_search(self.current_node, self.net, game, self.n_simulations, self.C_puct, self.dirichlet_alpha, self.training)
        self.current_node = self.current_node.edges[move]
        
        return move
    
    def __repr__(self):
        return f'<AlphaNet agent: {self.name}>'


class NetAgent:
    def __init__(self, name='NN only agent'):
        self.name = name
        self.net = AlphaNet().to(device)
    
    def load_model(self, checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=("cuda" if torch.cuda.is_available() else "cpu"))
        state_dict = checkpoint['model_state_dict']
        self.net.load_state_dict(state_dict)
    
    def get_next_move(self, game):
        P = self.net(game.state)[1].detach().squeeze().cpu().numpy()
        action = np.argmax(P[game.valid_actions])
        return action

    def __repr__(self):
        return f'<MCTS agent: {self.name}>'


class MCTSAgent:
    def __init__(self, n_simulations=100, name='MCTS only agent'):
        self.n_simulations = n_simulations
        self.name = name
        self.current_node = None

    def rollout(leaf, game):

        for sim in self.n_simulations:
            game_copy = copy.deepcopy(game)

            while not game_copy.outcome:
                random_action = random.choice(game_copy.valid_actions)
                game_copy.make_move(random_action)
            
            if game.outcome == leaf.player_id:
                outcome = 1
            elif game.outcome == 'tie':
                outcome = 0
            else: 
                outcome = -1

            backup(node, outcome)

    def get_next_move(game):
        if len(game.history) < 2:
            self.current_node = Node(game.state, game.player_turn, parent=None)
        else:
            opponent_move = game.history[-1]            
            self.current_node = self.current_node.edges[opponent_move]
        
        leaf = select_leaf(self.current_node, game)
        rollout(leaf, game)
        action = select_action(self.current_node, training=False)
        self.current_node = self.current_node.edges[action]
        
        return action

    def __repr__(self):
        return f'<MCTS agent>'

