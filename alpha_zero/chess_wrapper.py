from __future__ import annotations

from typing import Optional
import chess

# Constants to match tictactoe/generic interface
WHITE = True
BLACK = False
DRAW = "1/2-1/2"

class Move:
    def __init__(self, uci: str) -> None:
        self._uci: str = uci

    def uci(self) -> str:
        return self._uci
    
    def __str__(self) -> str:
        return self._uci
    
    def __repr__(self) -> str:
        return f"Move({self._uci})"
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Move):
            return self._uci == other._uci
        if isinstance(other, chess.Move):
            return self._uci == other.uci()
        return False

    def __hash__(self) -> int:
        return hash(self._uci)

class Board:
    def __init__(self, board: Optional[chess.Board] = None) -> None:
        if board is None:
            self.board: chess.Board = chess.Board()
        else:
            self.board = board

    def copy(self) -> Board:
        return Board(self.board.copy())

    def push(self, move: Move | chess.Move | str) -> None:
        # Support both Move wrapper, chess.Move, and UCI string
        if isinstance(move, Move):
            self.board.push_uci(move.uci())
        elif isinstance(move, str):
            self.board.push_uci(move)
        else:
            self.board.push(move)

    def push_uci(self, uci: str) -> None:
        self.board.push_uci(uci)

    def pop(self) -> chess.Move:
        return self.board.pop()

    @property
    def move_stack(self) -> list[chess.Move]:
        return self.board.move_stack

    @property
    def legal_moves(self) -> list[Move]:
        # Return list of Move wrappers
        return [Move(m.uci()) for m in self.board.legal_moves]

    def is_game_over(self) -> bool:
        return self.board.is_game_over()
    
    def result(self) -> str:
        return self.board.result()

    @property
    def turn(self) -> bool:
        return self.board.turn # chess.WHITE is True, chess.BLACK is False. Matches our constant.

    def piece_at(self, square: int) -> Optional[chess.Piece]:
        return self.board.piece_at(square)
    
    def __str__(self) -> str:
        return str(self.board)
