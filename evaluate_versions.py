import argparse
import os

from agents import AlphaAgent
from game.connect4 import Connect4
from evaluate import evaluate
from utils.logger import get_logger


parser = argparse.ArgumentParser()
parser.add_argument("--v1", default='', help='Checkpoint file for v1.')
parser.add_argument("--v2", default='', help='Checkpoint file for v2.')
parser.add_argument("--log_file", default='', help='Eval log file for models.')
parser.add_argument("--n_sims", default=1, type=int, help='number of simulations per move.', )
parser.add_argument("--n_episodes", default=1, type=int, help='number of games.')
ARGS = parser.parse_args()

logger = get_logger(__name__, ARGS.log_file)
logger.info(f'Evaluating {ARGS.v1}  VS. {ARGS.v2} for {ARGS.n_episodes} games with {ARGS.n_sims} simulations.')

v1 = AlphaAgent(n_simulations=ARGS.n_sims, training=False, name=ARGS.v1)
v2 = AlphaAgent(n_simulations=ARGS.n_sims, training=False, name=ARGS.v2)

v1.load_model(ARGS.v1)
v2.load_model(ARGS.v2)

print('loaded models')

totals, percents = evaluate(Connect4, v1, v2, ARGS.n_episodes)

logger.info(f'Match percentages: {percents}\n')
print(percents)

print('Done')
