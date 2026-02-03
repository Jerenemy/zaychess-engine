from __future__ import annotations

from . import chess_wrapper as cw
from . import tictactoe as ttt
from .utils import (
    ChessActionConverter,
    TicTacToeActionConverter,
    chess_board_to_tensor,
    tictactoe_board_to_tensor,
)
from typing import Protocol, Iterable, Optional, runtime_checkable
import numpy as np

@runtime_checkable
class GameAdapter(Protocol):
    # Core metadata
    @property
    def action_space_size(self) -> int: ...
    @property
    def input_shape(self) -> tuple[int, int, int]: ...

    # State lifecycle
    def new_board(self): ...
    def copy_board(self, board): ...

    # Moves
    def legal_moves(self, board) -> Iterable: ...
    def push(self, board, move) -> None: ...

    # Terminal
    def is_terminal(self, board) -> bool: ...
    def result(self, board) -> str: ...
    def terminal_value(self, board) -> float: ...

    # Encoding/decoding
    def encode_move(self, move_uci: str) -> Optional[int]: ...
    def decode_move(self, idx: int) -> Optional[str]: ...
    def policy_to_tensor(self, policy_dict: dict[str, float]) -> np.ndarray: ...

    # Tensorization
    def board_to_tensor(self, board) -> np.ndarray: ...


# alpha_zero/game_adapter.py




class ChessAdapter:
    def __init__(self) -> None:
        self.converter: ChessActionConverter = ChessActionConverter()

    @property
    def action_space_size(self) -> int:
        return self.converter.action_space_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (12, 8, 8)

    def new_board(self) -> cw.Board:
        return cw.Board()

    def copy_board(self, board: cw.Board) -> cw.Board:
        return board.copy()

    def legal_moves(self, board: cw.Board) -> Iterable[cw.Move]:
        return board.legal_moves

    def push(self, board: cw.Board, move: cw.Move | str) -> None:
        board.push(move)

    def is_terminal(self, board: cw.Board) -> bool:
        return board.is_game_over()

    def result(self, board: cw.Board) -> str:
        return board.result()

    def _current_player_id(self, board: cw.Board) -> int:
        # +1 for White, -1 for Black
        return 1 if board.turn == cw.WHITE else -1

    def terminal_value(self, board: cw.Board) -> float:
        result_str = board.result()
        if result_str == "1/2-1/2" or result_str == "*":
            return 0.0
        winner_id = 1 if result_str == "1-0" else -1
        return 1.0 if self._current_player_id(board) == winner_id else -1.0

    def encode_move(self, move_uci: str) -> Optional[int]:
        return self.converter.encode(move_uci)

    def decode_move(self, idx: int) -> Optional[str]:
        return self.converter.decode(idx)

    def policy_to_tensor(self, policy_dict: dict[str, float]) -> np.ndarray:
        return self.converter.policy_to_tensor(policy_dict)

    def board_to_tensor(self, board: cw.Board) -> np.ndarray:
        return chess_board_to_tensor(board)


class TicTacToeAdapter:
    def __init__(self) -> None:
        self.converter: TicTacToeActionConverter = TicTacToeActionConverter()

    @property
    def action_space_size(self) -> int:
        return self.converter.action_space_size

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return (2, 3, 3)

    def new_board(self) -> ttt.Board:
        return ttt.Board()

    def copy_board(self, board: ttt.Board) -> ttt.Board:
        return board.copy()

    def legal_moves(self, board: ttt.Board) -> Iterable[ttt.Move]:
        return board.legal_moves

    def push(self, board: ttt.Board, move: ttt.Move | str | int) -> None:
        board.push(move)

    def is_terminal(self, board: ttt.Board) -> bool:
        return board.is_game_over()

    def result(self, board: ttt.Board) -> str:
        return board.result()

    def _current_player_id(self, board: ttt.Board) -> int:
        # board.turn can be True/False or +/-1, normalize to +/-1
        return 1 if board.turn in (True, 1) else -1

    def terminal_value(self, board: ttt.Board) -> float:
        result_str = board.result()
        if result_str == "1/2-1/2" or result_str == "*":
            return 0.0
        winner_id = 1 if result_str == "1-0" else -1
        return 1.0 if self._current_player_id(board) == winner_id else -1.0

    def encode_move(self, move_uci: str) -> Optional[int]:
        return self.converter.encode(move_uci)

    def decode_move(self, idx: int) -> Optional[str]:
        return self.converter.decode(idx)

    def policy_to_tensor(self, policy_dict: dict[str, float]) -> np.ndarray:
        return self.converter.policy_to_tensor(policy_dict)

    def board_to_tensor(self, board: ttt.Board) -> np.ndarray:
        return tictactoe_board_to_tensor(board)
