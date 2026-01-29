import numpy as np

WHITE = True
BLACK = False
DRAW = "1/2-1/2"

class Move:
    def __init__(self, uci):
        self._uci = uci

    def uci(self):
        return self._uci
    
    def __str__(self):
        return self._uci
    
    def __repr__(self):
        return f"Move({self._uci})"
    
    def __eq__(self, other):
        return isinstance(other, Move) and self._uci == other._uci

    def __hash__(self):
        return hash(self._uci)

class Board:
    def __init__(self, board_array=None, turn=WHITE):
        # 3x3 board
        # 0 represents empty, 1 is X (White), -1 is O (Black)
        if board_array is None:
            self.board = np.zeros((3, 3), dtype=int)
        else:
            self.board = board_array
        self.turn = turn
        self._result = None
        self._game_over = False
        self.move_stack = []

    def copy(self):
        new_board = Board(self.board.copy(), self.turn)
        new_board._result = self._result
        new_board._game_over = self._game_over
        new_board.move_stack = self.move_stack.copy()
        return new_board

    def push(self, move):
        # Move can be integer (0-8) or string "0"-"8" or Move object
        if isinstance(move, Move):
            move_uci = move.uci()
        elif isinstance(move, str):
            move_uci = move
        else:
            move_uci = str(move)
            
        idx = int(move_uci)
        row, col = divmod(idx, 3)
        
        if self.board[row, col] != 0:
            raise ValueError(f"Illegal move: {move_uci}")

        val = 1 if self.turn == WHITE else -1
        self.board[row, col] = val
        self.move_stack.append(move)
        
        # Check for win/loss/draw
        self._check_status(row, col)
        
        self.turn = -self.turn

    def push_uci(self, uci):
        self.push(Move(uci))

    def pop(self):
        if not self.move_stack:
            return
        
        move = self.move_stack.pop()
        idx = int(move.uci())
        row, col = divmod(idx, 3)
        
        self.board[row, col] = 0
        self.turn = not self.turn # Revert turn
        
        # Invalidate cached status
        self._result = None
        self._game_over = False 
        # Ideally we'd re-check or store history of status, but for MCTS we mainly push copies.
        # But if we really need pop for MCTS traversal on single state (not recommended), we'd need to re-scan board.
        # For AlphaZero implementations usually we copy states, so pop might not be critical, but let's be safe:
        # Re-evaluating status from scratch is cheap for TTT.
        # Actually, self._check_status assumes Just Made Move. 
        # If we pop, we are back to Unfinished (or previous state). 
        # Since TTT cannot have captures or irreversible changes other than placing a piece:
        # If it was game over, now it's not (unless we popped a move after game over...?)
        # Let's just say we assume valid popping.

    @property
    def legal_moves(self):
        if self.is_game_over():
            return []
        
        moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r, c] == 0:
                    # UCI for TTT: "0"..."8"
                    idx = r * 3 + c
                    moves.append(Move(str(idx)))
        return moves

    def is_game_over(self):
        # We check this lazily or aggressively?
        # _check_status sets it. 
        # But if we just created board from array?
        if self._game_over:
            return True
        # If not set, maybe check? but MCTS uses push primarily.
        return False
    
    def result(self):
        # Return "1-0", "0-1", "1/2-1/2" or "*"
        if self._result is None:
            return "*"
        return self._result

    def _check_status(self, last_r, last_c):
        # Check row
        if np.all(self.board[last_r, :] == self.board[last_r, 0]):
            self._set_winner(self.board[last_r, 0])
            return
        # Check col
        if np.all(self.board[:, last_c] == self.board[0, last_c]):
            self._set_winner(self.board[0, last_c])
            return
        # Check diag
        if last_r == last_c:
            if np.all(np.diag(self.board) == self.board[0, 0]):
                self._set_winner(self.board[0, 0])
                return
        # Check anti-diag
        if last_r + last_c == 2:
            if np.all(np.diag(np.fliplr(self.board)) == self.board[0, 2]):
                self._set_winner(self.board[0, 2])
                return
        
        # Check draw
        if np.all(self.board != 0):
            self._result = "1/2-1/2"
            self._game_over = True
            return

    def _set_winner(self, player_val):
        self._game_over = True
        # player_val is the board value: 1 (White/X) or -1 (Black/O)
        # But we need to set result based on who that corresponds to.
        # WHITE is True, BLACK is False.
        # Value 1 -> White, -1 -> Black
        
        if player_val == 1:
            self._result = "1-0"
        elif player_val == -1:
            self._result = "0-1"
        else:
            self._result = "1/2-1/2"

    def __str__(self):
        s = ""
        symbols = {0: ".", 1: "X", -1: "O"}
        for r in range(3):
            for c in range(3):
                s += symbols[self.board[r, c]] + " "
            s += "\n"
        return s
