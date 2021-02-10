import random

def play_game(game, agent1, agent2, shuffle_order=True, verbose=False):
    """
    Play 1 game of given game instance. Verbose option for command line play.

    Args:
        - game: game instance
        - agent1, agent2: agents that implement get_next_move() and has a val for name property.
    
    Returns:
        - winning agent name (e.g. agent1.name) or "tie".
    """
    if shuffle_order: 
        turn = random.choice((agent1.name, agent2.name))
    else: 
        turn = agent1.name

    if turn == agent1.name:
        agent1_id = 1
        agent2_id = 2
    else:
        agent1_id = 2
        agent2_id = 1
    
    if verbose:
        print('New game!')
    
    while not game.outcome:
        if turn == agent1.name:
            if verbose:
                print(game)
                print(f"{agent1.name}'s turn.")
            
            move = agent1.get_next_move(game)
            if move not in game.valid_actions:
                move = random.choice(game.valid_actions)
                print(f'Illegal move. Random move ({move}) chosen instead.')
            game.make_move(move)
            turn = agent2.name

        elif turn == agent2.name:
            if verbose:
                print(game)
                print(f"{agent2.name}'s turn.")

            move = agent2.get_next_move(game)    
            if move not in game.valid_actions:
                print(f'Illegal move. Random move ({move}) chosen instead.')
                move = random.choice(game.valid_actions)
            game.make_move(move)
            turn = agent1.name

    if game.outcome == agent1_id:
        outcome = agent1.name
    elif game.outcome == agent2_id:
        outcome = agent2.name
    else:
        outcome = 'tie'

    if verbose:
        print(game)
        print('\nGame Over!')
        if outcome == 'tie':
            print('Tie game!')
        else:
            print(f'Winner: {outcome}!')
    
    return outcome

