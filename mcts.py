import copy
import logging
import math
import numpy as np
import random

import torch

from alpha_net import AlphaNet
from game.connect4 import Connect4
from utils.logger import get_logger



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Node:
    node_id = 0
    def __init__(self, state: np.array, player_id: int, parent=None):
        self.id = Node.node_id
        self.player_id = player_id # necessary to interpret game outcome for value assignment
        self.state = state  # game state
        self.parent = parent
        self.edges = {}  # {k=action: v=child node}
        self.W = 0  # sum of value derived below node
        self.N = 0  # visit count
        self.P = None  # prior probabilities for action. {k=action: v=prob}
        Node.node_id += 1
    
    @property
    def Q(self):
        """Average value derived below node."""
        return self.W / self.N if self.N > 0 else 0

    @property
    def explored(self):
        """Boolean for whether node has been explored. False for all terminal nodes."""
        return True if self.edges else False

    @ property
    def Pi(self):
        """Improved action probabilities derived from MCTS."""
        policy_probs = np.zeros(7) # adjust to game.actions if want to use for other board dims or games
        for action in range(len(policy_probs)):  
            if action in self.edges:
                policy_probs[action] = (self.edges[action].N / self.N)
        return policy_probs
    
    def __repr__(self):
        return f'MCTS Node for player {self.player_id}, ID: {self.id}, Q: {self.Q:.3f}, W: {self.W:.3f}, N: {self.N}, P: {self.P}'

def select_leaf(node: Node, game: Connect4,  C_puct=1.0) -> Node:
    """
    Find a leaf node by recursively traversing the game tree.
    
    Take actions that maximize Q + U where U is a variant of the UCT
    algorithm that controls exploration. We use the negative Q of child
    nodes to account for the switch in perspective from current node to
    child node. U is large for nodes with small N and high prior probabilities
    and asympotically selects paths with high Q vals. A leaf node indicates a
    terminal or unexplored state.
    """

    # base case: return discovered leaf node.
    if not node.explored:
        return node
    
    # recursively take actions that maximize Q + U until a leaf node is found.
    highest_score = -float('inf')
    next_action = None
    for action in node.edges:
        Q = -node.edges[action].Q  
        U = C_puct * node.P[action] * (np.sqrt(node.N) / (1 + node.edges[action].N))
        if Q + U > highest_score:
            highest_score = Q + U
            next_action = action
    
    game.make_move(next_action)
    next_node = node.edges[next_action]

    return select_leaf(next_node, game)

def prior_action_probs(state: np.array, net: AlphaNet, game: Connect4, dirichlet_alpha: float) -> dict:
    """Return dict of prior action probabilities derived from net policy head."""
    prior_probs = net(state.to(device))[1].detach().cpu().squeeze().numpy()
    dirichlet_noise = np.zeros(7)
    dirichlet_vals = np.random.dirichlet([dirichlet_alpha] * len(game.valid_actions))
    dirichlet_noise[game.valid_actions] = dirichlet_vals
    prior_probs = (0.75 * prior_probs) + (0.25 * dirichlet_noise)
    prior_probs[game.invalid_actions] = 0.0  # mask illegal actions

    # make prior probabilties of valid actions sum to 1
    prior_probs[game.valid_actions] = prior_probs[game.valid_actions] / prior_probs.sum()
    prior_probs = dict(enumerate(prior_probs))

    return prior_probs

def backup(node: Node, V: float) -> None:
    """Recursively update the nodes along the path taken to reach given node"""
    node.N += 1
    node.W += V
    if node.parent:
        backup(node.parent, -V)

def process_leaf(leaf: Node, net: AlphaNet, game: Connect4, dirichlet_alpha: float) -> None:
    """
    Get value for leaf state, initialize child nodes, update tree.

    Query NN for value and action probabilities for leaf state.
    Action probabilities are assigned as prior probabilities for all 
    leaf node edges. All possible edges and resulting nodes of the 
    leaf node are initilized. Traverse backwards up the tree to update
    all nodes along the path to the leaf node.
    """
    if game.outcome:  # leaf is a terminal node
        if game.outcome == leaf.player_id: 
            V = game.outcome
        elif game.outcome == 'tie': 
            V = 0
        else: 
            V = -1
        backup(leaf, V)
    
    V = net(leaf.state.to(device))[0].item()
    leaf.P = prior_action_probs(leaf.state, net, game, dirichlet_alpha)

    # initialize all possible edges and resulting child nodes.
    for action in game.valid_actions:
        game_copy = copy.deepcopy(game)
        game_copy.make_move(action)
        new_state = game_copy.state
        child_node = Node(new_state, parent=leaf, player_id=game_copy.player_turn)
        leaf.edges[action] = child_node

    backup(leaf, V)

def select_action(node, training: bool) -> int:
    """
    Select an action after MCTS simulations.

    If training, select an action from the current state proportional to  
    visit count. Otherwise, select the most visited action. Selecting action
    proportional to visits mimics a constant temperature setting of 1.0 in
    regards to AlphaZero action selection equatio: 
    Pi(a|s) = (N(s,a)**(1/temp)) / (N(s,b)**(1/temp)) where N(s,b) is sum
    of visits to each possible edge.
    """

    # logger.info(f'Selecting action from: {node}')

    if not training:
        most_visited = max(node.edges.keys(), key=lambda x: node.edges[x].N)
        return most_visited

    actions = list(node.edges.keys())
    next_action = random.choices(actions, node.Pi[actions])[0]
    return next_action

def mcts_search(root: Node, net: AlphaNet, game: Connect4, n_simulations: int,
                        C_puct: float, dirichlet_alpha: float, training: bool) -> int:
    """Return selected action after executing given number of MCTS simulations from root node"""
    for simulation in range(n_simulations):
        game_copy = copy.deepcopy(game)
        leaf = select_leaf(root, game_copy, C_puct)
        process_leaf(leaf, net, game_copy, dirichlet_alpha)  
        
    action = select_action(root, training)

    return action

def mcts_self_play(net: AlphaNet, game: Connect4, n_simulations: int, C_puct: float, dirichlet_alpha: float) -> tuple:
    """
    Generate training data via self-play. Returns list of (state, Pi, Z) tuples.
    Pi: improved action probabilities resulting from MCTS.
    Z: game outcome with value in [-1, 0, 1] for loss, tie, draw.
    """
    states, Pis, Zs = [],[],[]
    current_node = Node(game.state, parent=None, player_id=game.player_turn)

    while not game.outcome:

        action = mcts_search(current_node, net, game, n_simulations, C_puct, dirichlet_alpha, training=True)
        states.append(game.state)
        Pis.append(current_node.Pi)
        Zs.append(0)  # placeholder with value for a tie game.
        
        game.make_move(action)
        current_node = current_node.edges[action]
    
    p1_move_count = len(Zs[::2])
    p2_move_count = len(Zs[1::2])
    
    if game.outcome == 1:
        Zs[::2] = [1] * p1_move_count
        Zs[1::2] = [-1] * p2_move_count
    elif game.outcome == 2: 
        Zs[::2] = [-1] * p1_move_count
        Zs[1::2] = [1] * p2_move_count
    

    return states, Pis, Zs

    