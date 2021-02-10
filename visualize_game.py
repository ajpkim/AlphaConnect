import pickle
import time

def read_game_history(game_history_file):
    """Return list of lists holding game history with actions represented as ints."""
    with open(game_history_file, 'rb') as f:
        game_history = pickle.load(f)
    return game_history

def visualize_game(game, game_history):
    """Replay given game history for given game."""
    print('Replaying game history...\n')
    print(game)
    time.sleep(2.5)

    for action in game_history:

        print(f'\nPlayer {game.player_turn} took action {action}.')
        game.make_move(action)
        print(game)
        time.sleep(2)

    if game.outcome == 'tie':
        print('\nTie game')
    else:
        print(f'\nWinner: Player {game.outcome}')

if __name__ == '__main__':
    import argparse
    from game.connect4 import Connect4

    parser = argparse.ArgumentParser()
    parser.add_argument('--game_history_file')
    parser.add_argument('--game_number', type=int, default=0)
    ARGS = parser.parse_args()
    game_histories = read_game_history(ARGS.game_history_file)
    visualize_game(Connect4(), game_histories[ARGS.game_number])
    # import pdb
    # breakpoint()
    
    
