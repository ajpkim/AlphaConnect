import copy
import numpy as np
import random

from alpha_net import AlphaNet
from game.connect4 import Connect4

### NEED TO HANDLE FOR POV IE TAKE ARGMAX OF - Q + U, ETC...
### ---> can just create 2 separate trees and just take negative argmaxs
## How can i assign v to game outcome without tracking player turn and whatnot?
# Having a dedicated Tree datastructure will also allow me to query tree with a state to
# retrieve dataset info easily
# ---> Tree can just be a dict of Nodes

### I can use args and config arguments since these will be called from main script

class Node():
    id_num = 0
    def __init__(self, state, parent=None):
        self.id = Node.id_num
        self.state = state
        self.parent = parent
        self.edges = {}  # keys=action, values=child node
        self.terminal = False
        self.W = 0
        self.N = 0
        self.P = None 
        Node.id_num += 1

    @property
    def explored(self):
        return True if self.edges else False

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0
    
    @ property
    def Pi(self):
        policy_probs = []
        for action in range(7):
            if action in self.edges:
                policy_probs.append(self.edges[action].N / self.N)
            else:
                policy_probs.append(0)
        return policy_probs
    
    def __repr__(self):
        return f'MCTS Node id: {self.id}, Q: {self.Q:.3f}, W: {self.W:.3f}, N: {self.N}, P: {self.P}'

def prior_action_probs(state: np.array, net: AlphaNet, game: Connect4) -> dict:
    """
    Query NN policy head for action probabilities. Assign 0 to invalid moves.
    Return dict of action keys and action probabilty values.
    """
    prior_probs = net(state)[1].detach().squeeze().numpy()
    
    # ADD DIERLECHT NOISE 
    
    
    prior_probs[game.invalid_actions] = 0.000
    prior_probs = dict(enumerate(prior_probs))
    return prior_probs

def select_leaf(node: Node, C_puct=1.0) -> Node:
    """
    Find a leaf node by recursively traversing the game tree by taking actions
    that maximize Q + U where U is a variant of the PUCT algorithm that 
    controls exploration. U is large for nodes with small N and high prior
    probabilities and asympotically selects paths with high Q vals. A leaf
    node is either a terminal or unexplored state.
    """

    # base case: return leaf node.
    if not node.explored:
        return node
    
    # recursively take actions that maximize Q + U until a leaf node is found.
    highest_score = -float('inf')
    next_node = None
    for action in node.edges:
        Q = node.edges[action].Q
        U = C_puct * node.P[action] * (np.sqrt(node.N) / (1 + node.edges[action].N))
        if Q + U > highest_score:
            highest_score = Q + U
            next_node = node.edges[action]
    
    return select_leaf(next_node)

def process_leaf(leaf: Node, net: AlphaNet, game: Connect4):
    """
    Query NN for value and action probabilities for leaf state.
    Value is backpropogated up the tree. Action probabilities are assigned
    as prior probabilities for all leaf node edges. All possible edges and
    resulting nodes of the leaf node are initilized. Traverse backwards up 
    the tree to update all nodes along the path to the leaf node.    
    """

    if game.outcome:  # leaf is a terminal node
        # GET OUTCOME BASED ON POV
        # V = ???
        backup(leaf, V)

    V = net(leaf.state)[0].item()

    leaf.P = prior_action_probs(leaf.state, net, game)

    # initialize all possible edges and resulting child nodes.
    ### CONSIDER CREATING METHODS TO UNDO MOST RECENT MOVE IF THIS IS TOO INEFFICIENT...
    for action in game.valid_actions:
        game_copy = copy.deepcopy(game)
        game_copy.make_move(action)
        new_state = game_copy.state
        child_node = Node(new_state, parent=leaf)
        leaf.edges[action] = child_node
    
    backup(leaf, V)

def backup(node: Node, V: float) -> None:
    "Recursively update the nodes along the path taken to reach given node"
    
    node.N += 1
    node.W += V
    if node.parent:
        backup(node.parent)

def select_action(node, game, training=True):
    """
    If training, select an action from the current state proportional to  
    visit count. Otherwise, select the most visited action.         
    """

    if not training:
        most_visited = max(node.edges.keys(), key=lambda x: node.edges[x].N)
        return most_visited

    actions = list(node.edges.keys())
    next_action = random.choices(actions, node.Pi[actions])[0]

    return next_action
###################################
    # visit_counts = np.array([node.edges[action].N for action in node.edges])
    # visit_probs = visit_counts / visit_counts.sum()
    ###################################
    # probs = {}
    # for action in node.edges:
    #     probs[actions] = node.edges[action].N / node.N
    
    # next_action = random.choices(list(probs.keys(), list(probs.values())))[0]
###################################
    

def run_simulations(root: Node, net: AlphaNet, game: Connect4, n_simulations: int) -> None:
    
    for simulation in range(n_simulations):
        leaf = select_leaf(root, net)
        process_leaf(leaf, net, game)
    

def mcts_self_play(net: AlphaNet, n_simulations=600, C_puct=1.0):
    "Generate training data via self-play. Returns list of (state, Pi, Z) tuples."
    
    states, Pis, Zs = [],[],[]
    game = Connect4()
    curr_node = Node(game.state, parent=None)

    while not game.outcome:
        run_simulations(root, net, game, n_simulations)
        action = select_action(curr_node, game, training=True)
        game.make_move(action)
        
        pass