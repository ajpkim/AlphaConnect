import random

def play_game(game, agent_1, agent_2, shuffle_order=True, verbose=False):
    """
    Play 1 game of given game instance.

    Args:
        - game: game instance
        - agent_1, agent_2: game agents that implement get_next_move()
    
    Returns:
        - 1 if agent_1 wins, 2 if agent_2 wins, 'tie' if outcome is tie.
    """
    if shuffle_order: 
        turn = random.choice((agent_1.name, agent_2.name))
    else: 
        turn = agent_1.name

    if turn == agent_1.name:
        agent_1_id = 1
        agent_2_id = 2
    else:
        agent_1_id = 2
        agent_2_id = 1
    
    # game = Connect4()
    if verbose:
        print('New game!')
    
    while not game.outcome:

        if turn == agent_1.name:
            if verbose:
                print(game)
                print(f"{agent_1.name}'s turn.")
            
            move = agent_1.get_next_move(game)
            if move not in game.valid_actions:
                move = random.choice(game.valid_actions)
                print(f'Illegal move. Random move ({move}) chosen instead.')
            game.make_move(move)
            turn = agent_2.name

        elif turn == agent_2.name:
            if verbose:
                print(game)
                print(f"{agent_2.name}'s turn.")

            move = agent_2.get_next_move(game)    
            if move not in game.valid_actions:
                print(f'Illegal move. Random move ({move}) chosen instead.')
                move = random.choice(game.valid_actions)
            game.make_move(move)
            turn = agent_1.name

    if game.outcome == agent_1_id:
        outcome = agent_1.name
    elif game.outcome == agent_2_id:
        outcome = agent_2.name
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

