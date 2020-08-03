import argparse
import logging
import os
import pdb
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

from alpha_loss import AlphaLoss
from alpha_net import AlphaNet
from game.connect4 import Connect4
from utils.dot_dict import DotDict
from utils.logger import get_logger, setup_logger
from mcts import Node, mcts_self_play, mcts_search
from trainer import Trainer



parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='config/default.yaml', help='Configuration file location.')
parser.add_argument("--checkpoint_file", default='', help='checkpoint file to load model, memory, and training step count.')
parser.add_argument("--log_file", default='logs/new_model_log.log', help='log file location.')
# parser.add_argument("--dataset_file", default='', help='file with training data.')
# parser.add_argument("--save_model_file", default='models/new_model_in_training', help='file to training model to.')
ARGS = parser.parse_args()

with open(ARGS.config_file, mode='r') as f:
    config = yaml.safe_load(f)
    config = DotDict(config)


torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

setup_logger(ARGS.log_file)
logger = get_logger(__name__, ARGS.log_file)

logger.info('\n\n----------   NEW TRAINING SESSION   ----------\n')
logger.info(f'config file: {ARGS.config_file}')
logger.info(f'checkpoint file: {ARGS.checkpoint_file}')
logger.info(f'configs: {config}')




model_dir, model_iter = os.path.split(ARGS.checkpoint_file)
game_data_dir = model_dir + '/game_data'
checkpoint_dir = model_dir + '/checkpoints'

trainer = Trainer(config)
print(trainer)

# SEND NET TO GPU IF AVAIL


### loop for

for it in range(config.steps):
    game_data = mcts_self_play(net, Connect4(), config.n_simulations, config.C_puct)
    











if ARGS.checkpoint_file:
    logger.info('loading checkpoint...')
    trainer.load_checkpoint(ARGS.checkpoint_file)



while len(trainer.replay_buffer.memory) < 500:


