import copy
import logging
import numpy as np
import random

from alpha_net import AlphaNet
from logger import get_logger
from game.connect4 import Connect4


### ISSUE - SELECTING ILLEGAL ACTIONS IN SELECT LEAF SEARCH
# shouldn't be happening since illegal actions shouldnt have nodes seeded

### I can use args and config arguments since these will be called from main script

test_log = 'test_logs/mcts.log'
logger = get_logger(__name__, test_log)


class Node:
    def __init__(self, state: np.array, player_id: int, parent=None):
        self.player_id = player_id # necessary to interpret game outcome for value assignment
        self.state = state  # game state
        self.parent = parent
        self.edges = {}  # {k=action: v=child node}
        self.W = 0  # sum of value derived below node
        self.N = 0  # visit count
        self.P = None  # prior probabilities for action. {k=action: v=prob}

    @property
    def explored(self):
        """Boolean for whether node has been explored. False for all terminal nodes."""
        return True if self.edges else False

    @property
    def Q(self):
        """Average value derived below node."""
        return self.W / self.N if self.N > 0 else 0
    
    @ property
    def Pi(self):
        """Improved action probabilities derived from MCTS."""
        policy_probs = np.zeros(7) # adjust to game.actions if want to use for other board dims or games
        for action in range(len(policy_probs)):  
            if action in self.edges:
                policy_probs[action] = (self.edges[action].N / self.N)
        return policy_probs
    
    def __repr__(self):
        return f'MCTS Node for player {self.player_id}, Q: {self.Q:.3f}, W: {self.W:.3f}, N: {self.N}, P: {self.P}'

def select_leaf(node: Node, game: Connect4,  C_puct=1.0) -> Node:
    """
    Find a leaf node by recursively traversing the game tree.
    
    Take actions that maximize Q + U where U is a variant of the PUCT
    algorithm that controls exploration. We use the negative Q of child
    nodes to account for the switch in perspective from current node to
    child node. U is large for nodes with small N and high prior probabilities
    and asympotically selects paths with high Q vals. A leaf node indicates a
    terminal or unexplored state.
    """
    logger.info('SELECT LEAF')
    logger.info(node)
    logger.info(f'\n{game}')

    # base case: return leaf node.
    if not node.explored:
        logger.info('NOT EXPLORED')
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

def prior_action_probs(state: np.array, net: AlphaNet, game: Connect4, dirich_alpha=0.75) -> dict:
    """Return dict of prior action probabilities derived from net policy head."""
    prior_probs = net(state)[1].detach().squeeze().numpy()
    dirichlet_noise = np.random.dirichlet([dirich_alpha] * len(prior_probs))
    prior_probs += dirichlet_noise
    prior_probs[game.invalid_actions] = 0.000
    prior_probs = dict(enumerate(prior_probs))

    return prior_probs

def backup(node: Node, V: float) -> None:
    """Recursively update the nodes along the path taken to reach given node"""
    node.N += 1
    node.W += V
    if node.parent:
        backup(node.parent, -V)

def process_leaf(leaf: Node, net: AlphaNet, game: Connect4):
    """
    Get value for leaf state, initialize child nodes, update tree.

    Query NN for value and action probabilities for leaf state.
    Action probabilities are assigned as prior probabilities for all 
    leaf node edges. All possible edges and resulting nodes of the 
    leaf node are initilized. Traverse backwards up the tree to update
    all nodes along the path to the leaf node. Negative of value is
    backpropogated up the tree to account for the switch in perspective
    between parent and child nodes.
    """
    logger.info('PROCESSING')
    logger.info(leaf)

    if game.outcome:  # leaf is a terminal node
        if game.outcome == leaf.player_id: 
            V = game.outcome
        elif game.outcome == 'tie': 
            V = 0
        else: 
            V = -1
        backup(leaf, -V)

    V = net(leaf.state)[0].item()
    leaf.P = prior_action_probs(leaf.state, net, game)

    # initialize all possible edges and resulting child nodes.
    for action in game.valid_actions:
        game_copy = copy.deepcopy(game)
        game_copy.make_move(action)
        new_state = game_copy.state
        child_node = Node(new_state, parent=leaf, player_id=game_copy.player_turn)
        leaf.edges[action] = child_node

    backup(leaf, -V)

def select_action(node, training=True):
    """
    Select an action after MCTS simulations.

    If training, select an action from the current state proportional to  
    visit count. Otherwise, select the most visited action.         
    """
    if not training:
        most_visited = max(node.edges.keys(), key=lambda x: node.edges[x].N)
        return most_visited

    actions = list(node.edges.keys())
    next_action = random.choices(actions, node.Pi[actions])[0]

    return next_action

def run_simulations(root: Node, net: AlphaNet, game: Connect4, n_simulations: int) -> None:
    
    for simulation in range(n_simulations):
        game_copy = copy.deepcopy(game)
        leaf = select_leaf(root, game_copy)
        process_leaf(leaf, net, game_copy)  

def mcts_search(state: np.array) -> int:
    pass


def mcts_self_play(net: AlphaNet, game: Connect4, n_simulations=400, C_puct=1.0):
    """
    Generate training data via self-play. Returns list of (state, Pi, Z) tuples.
    Pi: improved action probabilities resulting from MCTS.
    Z: game outcome with value in [-1, 0, 1] for loss, tie, draw.
    """
    states, Pis, Zs = [],[],[]
    current_node = Node(game.state, parent=None)

    while not game.outcome:
        run_simulations(root, net, game, n_simulations)
        action = select_action(current_node, game, training=True)
        game.make_move(action)
        current_node = current_node.edges[action]
    
        # STORE DATA

        pass

    