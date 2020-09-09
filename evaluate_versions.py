import argparse
import os
import time
from pathlib import Path

from agents import AlphaAgent
from game.connect4 import Connect4
from evaluate import evaluate
from utils.logger import get_logger


def evaluate_versions(v1, v2, log_file, n_sims, n_episodes):

    start = time.perf_counter()

    v1_name = Path(v1).stem
    v2_name = Path(v2).stem

    logger = get_logger(__name__, log_file)
    logger.info(f'Evaluating {v1_name}  VS. {v2_name} with {n_sims} simulations for {n_episodes} games each as player 1.')

    agent1 = AlphaAgent(n_simulations=n_sims, training=False, name=v1_name)
    agent2 = AlphaAgent(n_simulations=n_sims, training=False, name=v2_name)

    agent1.load_model(v1)
    agent2.load_model(v2)

    print('loaded models')

    results  = evaluate(Connect4, agent1, agent2, n_episodes)

    logger.info(f'Match results: {results}\n')
    print(results)

    end = time.perf_counter()
    run_time = (end - start)/60
    print(f'Run time: {run_time} mins')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", default='', help='Checkpoint file for v1.')
    parser.add_argument("--v2", default='', help='Checkpoint file for v2.')
    parser.add_argument("--log_file", default='', help='Eval log file for models.')
    parser.add_argument("--n_sims", default=1, type=int, help='number of simulations per move.', )
    parser.add_argument("--n_episodes", default=1, type=int, help='number of games.')
    ARGS = parser.parse_args()
    kwargs = vars(ARGS)

    evaluate_versions(**kwargs)

    print('Done')
