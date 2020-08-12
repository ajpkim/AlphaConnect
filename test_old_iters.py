import copy
from datetime import datetime
import numpy as np
import os
import sys
import time
import pdb
import pickle
import yaml

from typing import List
from typing import Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F


from agents import AlphaAgent, HumanPlayer, RandomPlayer, NetAgent
from alpha_loss import AlphaLoss
from alpha_net import AlphaNet
from evaluate import evaluate
from game.connect4 import Connect4
from mcts import *
from game.play_game import play_game
from replay_buffer import ReplayBuffer
from trainer import Trainer
from utils.load_config import load_config
from utils.logger import get_logger
from visualize_game import read_game_history, visualize_game

log_file = 'm2/evaluations.log'
logger = get_logger(__name__, log_file)

n_sims = 200
n_episodes = 50

new_agent = AlphaAgent(n_simulations=n_sims, name='m2 6.8k', training=False)
old_agent = AlphaAgent(name='old agent', n_simulations=n_sims, training=False)

check_dir = 'm2/checkpoints/'
new_checkpoint_file = check_dir + 'step_6800.zip'

new_agent.load_model(new_checkpoint_file)
    

print(f'Evaluating m2 against old iterations for {n_episodes} with {n_sims} simulations')
logger.info(f'Evaluating m2 against old iterations for {n_episodes} with {n_sims} simulations')

for i in range(1,7):

    old_check = check_dir + f'step_{i}000.zip'
    old_agent.load_model(old_check)
    old_agent.name = f'm2 version {i}000'

    print(f'Evaluating m2 step 6.8 vs step {i}000')
    logger.info(f'Evaluating m2 step 6.8 vs step {i}000')

    tot, per = evaluate(Connect4, old_agent, new_agent, n_episodes)
    print(tot); print()
    logger.info(tot)


print('DONE')
