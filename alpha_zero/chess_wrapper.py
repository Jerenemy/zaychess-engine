import chess

# Constants to match tictactoe/generic interface
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
        if isinstance(other, Move):
            return self._uci == other._uci
        if isinstance(other, chess.Move):
            return self._uci == other.uci()
        return False

    def __hash__(self):
        return hash(self._uci)

class Board:
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def copy(self):
        return Board(self.board.copy())

    def push(self, move):
        # Support both Move wrapper, chess.Move, and UCI string
        if isinstance(move, Move):
            self.board.push_uci(move.uci())
        elif isinstance(move, str):
            self.board.push_uci(move)
        else:
            self.board.push(move)

    def push_uci(self, uci):
        self.board.push_uci(uci)

    def pop(self):
        return self.board.pop()

    @property
    def move_stack(self):
        return self.board.move_stack

    @property
    def legal_moves(self):
        # Return list of Move wrappers
        return [Move(m.uci()) for m in self.board.legal_moves]

    def is_game_over(self):
        return self.board.is_game_over()
    
    def result(self):
        return self.board.result()

    @property
    def turn(self):
        return self.board.turn # chess.WHITE is True, chess.BLACK is False. Matches our constant.

    def piece_at(self, square):
        return self.board.piece_at(square)
    
    def __str__(self):
        return str(self.board)
