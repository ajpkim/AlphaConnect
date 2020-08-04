import argparse
import logging
import os
import pdb
import sys
from pathlib import Path
import pickle
from datetime import datetime

import numpy as np
import torch
import yaml

from alpha_loss import AlphaLoss
from alpha_net import AlphaNet
from game.connect4 import Connect4
from utils.dot_dict import DotDict
from utils.logger import get_logger, setup_logger
from utils.load_config import load_config
from mcts import Node, mcts_self_play, mcts_search
from trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='config/default.yaml', help='Configuration file location.')
parser.add_argument("--model_dir", default='models/new_model', help='directory to save model, training, and game data')
parser.add_argument("--checkpoint_file", default='', help='checkpoint file to load model, memory, and training step count.')
parser.add_argument("--log_file", default='logs/new_model_log.log', help='log file location.')
ARGS = parser.parse_args()

config = load_config(ARGS.config_file)

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

setup_logger(ARGS.log_file)
logger = get_logger(__name__, ARGS.log_file)

logger.info('\n\n----------   NEW SESSION   ----------\n')
logger.info(f'config file: {ARGS.config_file}')
logger.info(f'model dir: {ARGS.model_dir}')
logger.info(f'checkpoint file: {ARGS.checkpoint_file}')
logger.info(f'configs: {config}')

game_history_dir = ARGS.model_dir + '/game_histories'
checkpoint_dir = ARGS.model_dir + '/checkpoints'
# training_data_dir = ARGS.model_dir + '/training_data'

for directory in [ARGS.model_dir, game_history_dir, checkpoint_dir]: #, training_data_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f'created dir: {directory}')


trainer = Trainer(config)

if ARGS.checkpoint_file:
    logger.info('loading checkpoint')
    trainer.load_checkpoint(ARGS.checkpoint_file)
else: 
    logger.info('saving initial model state')
    trainer.save_checkpoint(checkpoint_dir + '/initialization')

game_histories = []

if len(trainer.replay_buffer.memory) < 500:
    while len(trainer.replay_buffer.memory) < 500:
        logger.info('initializing replay buffer memory with self play')
        game_history = trainer.self_play()
        game_histories.append(game_history)
    logger.info('writing initial game histories')
    with open(game_history_dir + '/init_game_histories', mode='wb') as f:
        pickle.dump(game_histories, f)
    
for step in range(1, config.steps + 1):
    logger.info(f'STEP {step}')
    game_history = trainer.self_play()
    game_histories.append(game_history)
    trainer.learn()

    if step % 25 == 0:
        logger.info('checkpointing model')
        trainer.save_checkpoint(checkpoint_dir + f'/step_{trainer.training_step_count}')
        logger.info('writing game histories')
        with open(game_history_dir + (f'/games_{step-25}_{step}'), 'wb') as f:
            pickle.dump(game_histories, f)
        game_histories = []
