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
from evaluate_versions import evaluate_versions
from game.connect4 import Connect4
from utils.logger import get_logger, setup_logger
from utils.load_config import load_config
from mcts import Node, mcts_self_play, mcts_search
from trainer import Trainer


start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='config/default.yaml', help='Configuration file location.')
parser.add_argument("--model_dir", default='new_model', help='directory to save model, training, and game data')
parser.add_argument("--checkpoint_file", default='', help='checkpoint file to load model, optimizer, scheduler, training/self-play step count.')
parser.add_argument("--memory_file", default='', help='replay buffer memory file to load memory.')
parser.add_argument("--log_file", default='new_model_log.log', help='log file location.')

ARGS = parser.parse_args()
config = load_config(ARGS.config_file)

game_history_dir = ARGS.model_dir + '/game_history'
checkpoint_dir = ARGS.model_dir + '/checkpoints'
replay_memory_dir = ARGS.model_dir + '/replay_memory'

for directory in [ARGS.model_dir, game_history_dir, checkpoint_dir, replay_memory_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

setup_logger(ARGS.log_file)
logger = get_logger(__name__, ARGS.log_file)

logger.info('\n\n\n----------   NEW SESSION   ----------\n')
logger.info(f'config file: {ARGS.config_file}')
logger.info(f'model dir: {ARGS.model_dir}')
logger.info(f'checkpoint file: {ARGS.checkpoint_file}')
logger.info(f'memory file: {ARGS.memory_file}')
logger.info(f'configs: {config}')

trainer = Trainer(config)

if ARGS.checkpoint_file:
    logger.info('loading checkpoint')
    trainer.load_checkpoint(ARGS.checkpoint_file)
    logger.info('loading memory')
    trainer.load_replay_memory(ARGS.memory_file)
else: 
    logger.info('checkpointing initial model state')
    trainer.save_checkpoint(checkpoint_dir + '/initial_state')

game_histories = []

if len(trainer.replay_buffer.memory) < 3000:
    print('Initializing replay buffer')
    while len(trainer.replay_buffer.memory) < 3000:
        game_history = trainer.self_play()
        game_histories.append(game_history)
    logger.info('writing initial game histories')
    with open(game_history_dir + f'/games_0_{trainer.self_play_count}', mode='wb') as f:
        pickle.dump(game_histories, f)
    game_histories = []
    logger.info('writing initial replay memory')
    trainer.save_replay_memory(replay_memory_dir + '/init_memory')

print('Entering Training Cycle')    
for step in range(1, config.steps + 1):
    if step % 10 == 0:
        update = f'Cycle Step {step} | Total training steps {trainer.training_step_count} | Run time {datetime.now()-start_time}'
        logger.info(update)
        print(update)

    game_history = trainer.self_play()
    game_histories.append(game_history)
    trainer.learn()

    if step % config.checkpoint_freq == 0:
        print('checkpointing model')
        logger.info('checkpointing model')
        trainer.save_checkpoint(checkpoint_dir + f'/step_{trainer.training_step_count}')
        logger.info('writing game histories')
        with open(game_history_dir + (f'/games_{trainer.self_play_count - config.checkpoint_freq + 1}_{trainer.self_play_count}'), 'wb') as f:
            pickle.dump(game_histories, f)
        game_histories = []
        logger.info('writing replay memory')
        trainer.save_replay_memory(replay_memory_dir + f'/step_{trainer.training_step_count}_memory')


if config.eval:
    update = '\n\n----------------------EVALUATION MODE----------------------'
    logger.info(update); print(update)
    v1 = ARGS.checkpoint_file
    v2 = checkpoint_dir + f'/step_{trainer.training_step_count}'
    evaluate_versions(v1=v1, v2=v2, log_file=ARGS.log_file, n_sims=config.eval_sims, n_episodes=config.eval_episodes)


print(f'DONE | Run time {datetime.now()-start_time}')


