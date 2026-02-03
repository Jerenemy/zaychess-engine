from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence, Callable
import numpy as np
import chess

# --- Action Converter Interface ---

class ActionConverter(Protocol):
    def encode(self, move_uci: str) -> Optional[int]:
        ...
    def decode(self, idx: int) -> Optional[str]:
        ...
    def policy_to_tensor(self, policy_dict: Dict[str, float]) -> np.ndarray:
        ...
    @property
    def action_space_size(self) -> int:
        ...

# --- Chess Converter (Original) ---

class ChessActionConverter:
    def __init__(self) -> None:
        # 1. Define the 8 compass directions (dx, dy)
        self.queen_dirs = [
            (1, 0), (1, 1), (0, 1), (-1, 1), 
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        # 2. Define the 8 Knight jumps
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1), 
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # 3. Underpromotion types
        self.underpromo_dirs = [0, -1, 1] 
        self.underpromo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        # 4. BUILD THE LOOKUP TABLES
        self.move_to_id: dict[str, int] = {}
        self.id_to_move: dict[int, str] = {}
        self._build_mapping()

    def _build_mapping(self) -> None:
        counter = 0
        for from_sq in range(64):
            rank = chess.square_rank(from_sq)
            col = chess.square_file(from_sq)
            
            # Type 1: Queen Moves
            for d_rank, d_col in self.queen_dirs:
                for dist in range(1, 8):
                    to_rank = rank + (d_rank * dist)
                    to_col = col + (d_col * dist)
                    if 0 <= to_rank < 8 and 0 <= to_col < 8:
                        to_sq = chess.square(to_col, to_rank)
                        move = chess.Move(from_sq, to_sq)
                        self._add_move(move, counter)
                        
                        if (rank == 6 and to_rank == 7) or (rank == 1 and to_rank == 0):
                            move_q = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                            self._add_move(move_q, counter)
                    counter += 1

            # Type 2: Knight Moves
            for d_rank, d_col in self.knight_moves:
                to_rank = rank + d_rank
                to_col = col + d_col
                if 0 <= to_rank < 8 and 0 <= to_col < 8:
                    to_sq = chess.square(to_col, to_rank)
                    move = chess.Move(from_sq, to_sq)
                    self._add_move(move, counter)
                counter += 1

            # Type 3: Underpromotions
            for d_col in self.underpromo_dirs:
                for piece in self.underpromo_pieces:
                    if rank == 6: # White
                        to_rank = 7
                        to_col = col + d_col
                        if 0 <= to_col < 8:
                            to_sq = chess.square(to_col, to_rank)
                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._add_move(move, counter)
                    if rank == 1: # Black
                        to_rank = 0
                        to_col = col + d_col
                        if 0 <= to_col < 8:
                            to_sq = chess.square(to_col, to_rank)
                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._add_move(move, counter)
                    counter += 1

    def _add_move(self, move: chess.Move, idx: int) -> None:
        uci = move.uci()
        if uci not in self.move_to_id:
            self.move_to_id[uci] = idx
            is_promo = len(uci) == 5 and uci[-1] in 'qnbr'
            if idx not in self.id_to_move or is_promo:
                self.id_to_move[idx] = uci

    def encode(self, move_uci: str) -> Optional[int]:
        if hasattr(move_uci, 'uci'):
            move_uci = move_uci.uci()
        else:
            move_uci = str(move_uci)
        return self.move_to_id.get(move_uci, None)

    def decode(self, idx: int) -> Optional[str]:
        return self.id_to_move.get(idx, None)
    
    def policy_to_tensor(self, policy_dict: Dict[str, float]) -> np.ndarray:
        arr = np.zeros(4672, dtype=np.float32)
        total_visits = sum(policy_dict.values())
        if total_visits <= 0:
            return arr
        for uci, visits in policy_dict.items():
            idx = self.encode(uci)
            if idx is not None:
                arr[idx] = visits / total_visits
        return arr

    @property
    def action_space_size(self) -> int:
        return 4672

# --- TicTacToe Converter ---

class TicTacToeActionConverter:
    def __init__(self) -> None:
        self.size: int = 9 # Only 9 squares
        
    def encode(self, move_uci: str) -> Optional[int]:
        # '0' -> 0, '8' -> 8
        try:
            return int(move_uci)
        except ValueError:
            return None

    def decode(self, idx: int) -> Optional[str]:
        if 0 <= idx < 9:
            return str(idx)
        return None
    
    def policy_to_tensor(self, policy_dict: Dict[str, float]) -> np.ndarray:
        arr = np.zeros(9, dtype=np.float32)
        total_visits = sum(policy_dict.values())
        if total_visits <= 0:
            return arr
        for uci, visits in policy_dict.items():
            idx = self.encode(uci)
            if idx is not None:
                arr[idx] = visits / total_visits
        return arr

    @property
    def action_space_size(self) -> int:
        return 9

# --- Board to Tensor Functions ---

def chess_board_to_tensor(board: chess.Board) -> np.ndarray:
    if hasattr(board, 'board'):
        board = board.board
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            row = chess.square_rank(square) 
            col = chess.square_file(square)
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            tensor[channel, row, col] = 1.0
    return tensor

def tictactoe_board_to_tensor(board: Any) -> np.ndarray:
    # simple 3 channel: P1 pieces, P2 pieces, Turn
    # Or just 2 channels: Own pieces, Opponent pieces? 
    # Let's match typical alpha zero TTT: 3 planes:
    # 0: pieces of turn player
    # 1: pieces of opponent
    # 2: color (all 0 or all 1) ?? 
    # Let's keep it simple for now as 2 channels: White(1), Black(-1)
    
    tensor = np.zeros((2, 3, 3), dtype=np.float32)
    # board.board is 3x3 with 1, -1, 0
    
    # Channel 0: White (1)
    tensor[0] = (board.board == 1).astype(np.float32)
    # Channel 1: Black (-1)
    tensor[1] = (board.board == -1).astype(np.float32)
    
    return tensor


# --- Common Utils ---

def sample_next_move(
    move_probs: Sequence[float] | np.ndarray,
    legal_moves: Optional[Sequence[Any]] = None,
    temperature: float = 1.0,
    encode_move: Optional[Callable[[str], Optional[int]]] = None,
    decode_move: Optional[Callable[[int], Optional[str]]] = None,
) -> str:
    probs = np.asarray(move_probs, dtype=np.float64).flatten()

    if encode_move is None or decode_move is None:
        raise ValueError("encode_move and decode_move must be provided.")
    
    if legal_moves is not None:
        mask = np.zeros_like(probs)
        for move in legal_moves:
            # handle both chess.Move and TTT Move (which has uci())
            uci = move.uci() if hasattr(move, 'uci') else str(move)
            idx = encode_move(uci)
            if idx is not None:
                mask[idx] = 1.0
        probs *= mask
        
    total = probs.sum()
    if total <= 0:
        if legal_moves:
            return list(legal_moves)[0].uci()
        raise ValueError("No moves available to sample.")
    
    probs /= total 
    
    if temperature == 0:
        idx = int(np.argmax(probs))
        return decode_move(idx)
    else:
        probs = np.power(probs, 1.0 / temperature)
        probs /= probs.sum() 
        
    idx = int(np.random.choice(len(probs), p=probs))
    move = decode_move(idx)
    
    if move is None:
        return decode_move(int(np.argmax(probs)))
    return move

import sys
import psutil
import os

def check_memory(logger: Any, step_name: str) -> None:
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3) 
    logger.debug(f"MEM CHECK [{step_name}]: {mem_gb:.2f} GB")
    if mem_gb > 16: 
        logger.error("!!! MEMORY CRITICAL !!! Shutting down safely.")
        sys.exit(1)
