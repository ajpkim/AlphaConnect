import numpy as np
import torch

class Connect4:

    def __init__(self, rows=6, cols=7):
        self.rows = rows
        self.cols = cols
        self.actions = [i for i in range(cols)]
        self.board = np.zeros((rows, cols))
        self.history = []
        self.turns = 0
        self.outcome = None

    @property
    def player_turn(self):
        if self.turns % 2 == 0:
            return 1
        return 2

    @property
    def valid_actions(self):
        valid_actions = []
        for i in range(self.cols):
            if self.board[0, i] == 0.0:
                valid_actions.append(i)
        return valid_actions

    @property
    def invalid_actions(self):
        invalid_actions = []
        for i in range(self.cols):
            if self.board[(0, i)] != 0:
                invalid_actions.append(i)
        return invalid_actions

    @property
    def state(self):
        state = np.zeros((3, self.rows, self.cols))
        for row in range(self.rows):
            for col in range(self.cols):
                if self.board[(row, col)] == 0:
                    continue
                elif self.board[(row, col)] == 1:
                    state[(0, row, col)] = 1
                else:
                    state[(1, row, col)] = 1

        if self.turns % 2 == 0:
            state[2, :] = 1
        return torch.tensor(state, dtype=(torch.float32))

    def make_move(self, col: int):
        """Attempt to play piece in given col. Raise error if illegal move."""
        if col in self.valid_actions:
            self.history.append(col)
            row = self.rows - self.col_height(col) - 1
            self.board[row][col] = self.player_turn
            self.turns += 1
            self.check_winning_move(col)
        else:
            raise Exception(f"Illegal move. Attemped to play in column {col}. "\
                            f"Valid actions: {self.valid_actions}\n{self.board}")
                              
    def col_height(self, col):
        """Return number of game pieces played in given col"""
        col = self.board[:, col]
        col = np.flip(col, axis=0)
        if col[(-1)] != 0:
            return self.rows
        return np.where(col == 0)[0][0]

    def check_winning_move(self, col):
        """Check if most recent move (given by col) is a winning move. Update game outcome"""
        if self.turns < 7:
            return False

        if self.turns == self.rows * self.cols:  # tie
            self.outcome = 'tie'
            return False

        row = self.rows - 1 - max(0, self.col_height(col) - 1)
        player_id = self.board[row][col]
        # vertical win
        if row + 3 >= self.rows:
            pass  # not enough pieces in col for vertical win.
        elif (self.board[row+1][col] == player_id and
              self.board[row+2][col] == player_id and
              self.board[row+3][col] == player_id):

            self.outcome = player_id
            return True

        # horizontal win
        left_most = max(0, col-3)
        right_most = min(self.cols-1, col+3)

        for c in range(left_most, right_most - 2):
            if (self.board[row][c] == player_id and
                self.board[row][c+1] == player_id and
                self.board[row][c+2] == player_id and
                    self.board[row][c+3] == player_id):

                self.outcome = player_id
                return True

        # negative diagonal win
        k = col - row  # diagonal offset
        neg_diag = np.diag(self.board, k=k)

        if len(neg_diag) > 3:
            for i in range(len(neg_diag) - 3):
                if (neg_diag[i] == player_id and
                    neg_diag[i+1] == player_id and
                    neg_diag[i+2] == player_id and
                        neg_diag[i+3] == player_id):

                    self.outcome = player_id
                    return True

        # positive diagonal win
        flipped_board = np.flip(self.board, axis=0)
        flipped_row = (self.rows - 1) - row
        k = col - flipped_row
        pos_diag = np.diag(flipped_board, k=k)

        if len(pos_diag) > 3:
            for i in range(len(pos_diag) - 3):
                if (pos_diag[i] == player_id and
                    pos_diag[i+1] == player_id and
                    pos_diag[i+2] == player_id and
                        pos_diag[i+3] == player_id):

                    self.outcome = player_id
                    return True

        return False
    
    def reset(self):
        self.board = np.zeros((3,6,7))
        self.history = []
        self.turns = 0
        self.outcome = None

    def __repr__(self):
        """Return a string represetation of board state"""
        s = ''
        for row in self.board:
            s += str(row) + '\n'
        return s
