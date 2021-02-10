import argparse
import os

import pickle
from datetime import datetime

import numpy as np
import torch
# from torch.utils.tensorboard import SummaryWriter

# from evaluate_versions import evaluate_versions
from trainer import Trainer
from utils.logger import get_logger, setup_logger
from utils.load_config import load_config


start_time = datetime.now()

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='config/default.yaml', help='Configuration file location.')
parser.add_argument('--checkpoint_file', default='', help='checkpoint file to load model, optimizer, scheduler, training/self-play step count.')
parser.add_argument('--memory_file', default='', help='replay buffer memory file to load memory.')
parser.add_argument('--log_file', default='new_model_log.log', help='log file location.')
parser.add_argument('--steps', type=int, default=1000, help='Number of self-play games and training steps.')
parser.add_argument('--checkpoint_freq', type=int, default=250, help='Number of steps between each checkpoint.')
parser.add_argument('--update_freq', type=int, default=100, help='Frequency of step and time updates')

ARGS = parser.parse_args()
config = load_config(ARGS.config_file)

checkpoint_dir = './checkpoints'
model_dir = './models'
game_history_dir = './game_history'
replay_memory_dir = './replay_memory'

for directory in [checkpoint_dir, model_dir, game_history_dir, replay_memory_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

torch.manual_seed(config.random_seed)
np.random.seed(config.random_seed)

setup_logger(ARGS.log_file)
logger = get_logger(__name__, ARGS.log_file)

logger.info('\n\n\n----------   NEW TRAINING SESSION   ----------\n')
logger.info(f'checkpoint file: {ARGS.checkpoint_file}')
logger.info(f'memory file: {ARGS.memory_file}')
logger.info(f'config file: {ARGS.config_file}')
logger.info(f'config: {config}')

trainer = Trainer(config)

if ARGS.checkpoint_file and ARGS.memory_file:
    logger.info('loading checkpoint...')
    trainer.load_checkpoint(ARGS.checkpoint_file)
    logger.info('loading memory...')
    trainer.load_replay_memory(ARGS.memory_file)
else:
    logger.info('checkpointing initial state')
    trainer.save_checkpoint(checkpoint_dir + '/initial_checkpoint')

game_histories = []
if len(trainer.replay_buffer.memory) < config.memory_init_capacity:
    print('Initializing replay buffer')
    while len(trainer.replay_buffer.memory) < config.memory_init_capacity:
        game_history = trainer.self_play()
        game_histories.append(game_history)
    logger.info('writing initial game histories')
    with open(game_history_dir + f'/init_games', mode='wb') as f:
        pickle.dump(game_histories, f)
    game_histories = []
    logger.info('writing initial replay memory')
    trainer.save_replay_memory(replay_memory_dir + '/init_memory')

print('Entering Training Cycle')
for step in range(1, ARGS.steps + 1):
    if step % ARGS.update_freq == 0:
        update = f'Cycle Step {step} | Total training steps {trainer.training_steps + 1} | Run time {datetime.now()-start_time}'
        logger.info(update)
        print(update)

    game_history = trainer.self_play()
    game_histories.append(game_history)
    trainer.learn()

    if step % ARGS.checkpoint_freq == 0:
        msg = f'checkpointing model at step: {trainer.training_steps}'
        print(msg)
        logger.info(msg)

        trainer.save_checkpoint(checkpoint_dir + f'/step_{trainer.training_steps}')
        logger.info(f'writing game histories {trainer.training_steps - ARGS.checkpoint_freq + 1} - {trainer.training_steps}')
        with open(game_history_dir + (f'/games_{trainer.training_steps - ARGS.checkpoint_freq + 1}_to_{trainer.training_steps}'), 'wb') as f:
            pickle.dump(game_histories, f)
        game_histories = []
        logger.info('writing replay memory')
        trainer.save_replay_memory(replay_memory_dir + f'/step_{trainer.training_steps}_memory')

# if config.eval:
#     update = '\n\n----------------------EVALUATION MODE----------------------'
#     logger.info(update); print(update)
#     v1 = ARGS.checkpoint_file
#     v2 = checkpoint_dir + f'/step_{trainer.training_steps}'
#     evaluate_versions(v1=v1, v2=v2, log_file=ARGS.log_file, n_sims=config.eval_sims, n_episodes=config.eval_episodes)

# print(f'DONE | Run time {datetime.now()-start_time}')
