import pickle
import time

def read_game_history(game_histoy_file):
    """Return list of lists holding game history with actions represented as ints."""
    with open(game_histoy_file, 'rb') as f:
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
        print(f'Winner: Player {game.outcome}')
    