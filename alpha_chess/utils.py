import numpy as np
import chess

class ActionConverter:
    def __init__(self):
        # 1. Define the 8 compass directions (dx, dy)
        # N, NE, E, SE, S, SW, W, NW
        self.queen_dirs = [
            (1, 0), (1, 1), (0, 1), (-1, 1), 
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]
        
        # 2. Define the 8 Knight jumps
        self.knight_moves = [
            (2, 1), (1, 2), (-1, 2), (-2, 1), 
            (-2, -1), (-1, -2), (1, -2), (2, -1)
        ]
        
        # 3. Underpromotion types (Knight, Bishop, Rook)
        # Directions: Forward, Capture Left, Capture Right
        self.underpromo_dirs = [0, -1, 1] # Delta file
        self.underpromo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

        # 4. BUILD THE LOOKUP TABLES
        self.move_to_id = {}
        self.id_to_move = {}
        self._build_mapping()

    def _build_mapping(self):
        counter = 0
        
        # We iterate over every possible FROM square (0-63)
        for from_sq in range(64):
            rank = chess.square_rank(from_sq)
            col = chess.square_file(from_sq)
            
            # --- TYPE 1: QUEEN MOVES (56 planes) ---
            # 8 directions * 7 distances
            for d_rank, d_col in self.queen_dirs:
                for dist in range(1, 8):
                    # Calculate TO square
                    to_rank = rank + (d_rank * dist)
                    to_col = col + (d_col * dist)
                    
                    if 0 <= to_rank < 8 and 0 <= to_col < 8:
                        to_sq = chess.square(to_col, to_rank)
                        
                        # Create standard move (and Queen promotion)
                        # We map BOTH "e7e8" and "e7e8q" to this plane
                        move = chess.Move(from_sq, to_sq)
                        self._add_move(move, counter)
                        
                        # Handle Queen Promotion specifically (UCI uses 'q')
                        # Only add if it's a valid promotion move to avoid 'e2e4q'
                        if (rank == 6 and to_rank == 7) or (rank == 1 and to_rank == 0):
                            move_q = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
                            self._add_move(move_q, counter)
                        
                    counter += 1 # Plane used even if move is OOB (to keep shape consistent)

            # --- TYPE 2: KNIGHT MOVES (8 planes) ---
            for d_rank, d_col in self.knight_moves:
                to_rank = rank + d_rank
                to_col = col + d_col
                
                if 0 <= to_rank < 8 and 0 <= to_col < 8:
                    to_sq = chess.square(to_col, to_rank)
                    move = chess.Move(from_sq, to_sq)
                    self._add_move(move, counter)
                
                counter += 1

            # --- TYPE 3: UNDERPROMOTIONS (9 planes) ---
            # Only relevant for ranks 1->0 (White) and 6->7 (Black)? 
            # AlphaZero encodes them relative to square, regardless of validity.
            for d_col in self.underpromo_dirs: # -1, 0, 1
                for piece in self.underpromo_pieces:
                    # Direction is always "forward" in rank, but file changes
                    # We assume White 'forward' (+1 rank) for the map encoding usually
                    # But for generic mapping, we just check validity
                    
                    # Logic: Try pushing 'forward' relative to the square's perspective? 
                    # Simpler: Just map the vector.
                    
                    # NOTE: For simplicity in this script, we map specific moves
                    # valid only on promotion ranks.
                    
                    # White Promotion (Rank 6 -> 7)
                    if rank == 6:
                        to_rank = 7
                        to_col = col + d_col
                        if 0 <= to_col < 8:
                            to_sq = chess.square(to_col, to_rank)
                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._add_move(move, counter)

                    # Black Promotion (Rank 1 -> 0)
                    if rank == 1:
                        to_rank = 0
                        to_col = col + d_col
                        if 0 <= to_col < 8:
                            to_sq = chess.square(to_col, to_rank)
                            move = chess.Move(from_sq, to_sq, promotion=piece)
                            self._add_move(move, counter)
                            
                    counter += 1

    def _add_move(self, move, idx):
        # We store the UCI string for easy lookup
        uci = move.uci()
        if uci not in self.move_to_id:
            self.move_to_id[uci] = idx
            # If it's a promotion move (ends with q/n/b/r), or if idx not yet in id_to_move, store it
            # This ensures 'e7e8q' is stored instead of 'e7e8' if both map to same index.
            is_promo = len(uci) == 5 and uci[-1] in 'qnbr'
            if idx not in self.id_to_move or is_promo:
                self.id_to_move[idx] = uci

    def encode(self, move_uci: str) -> int:
        return self.move_to_id.get(move_uci, None)

    def decode(self, idx: int) -> str:
        return self.id_to_move.get(idx, None)
    
    def policy_to_tensor(self, policy_dict):
        """
        Converts a dictionary of {move_uci: count} to a 4672-length probability array.
        """
        arr = np.zeros(4672, dtype=np.float32)
        total_visits = sum(policy_dict.values())
        if total_visits <= 0:
            return arr
        
        for uci, visits in policy_dict.items():
            idx = self.encode(uci)
            if idx is not None:
                arr[idx] = visits / total_visits
                
        return arr


converter = ActionConverter()

# Samples a move UCI string from a policy distribution.
def sample_next_move(move_probs, legal_moves=None, temperature=1.0):
    probs = np.asarray(move_probs, dtype=np.float64).flatten()
    
    # 1. Mask illegal moves
    if legal_moves is not None:
        mask = np.zeros_like(probs)
        for move in legal_moves:
            idx = converter.encode(move.uci())
            if idx is not None:
                mask[idx] = 1.0
        probs *= mask
        
    total = probs.sum()
    if total <= 0:
        if legal_moves: # Fallback
             return list(legal_moves)[0].uci()
        raise ValueError("No moves available to sample.")
    
    probs /= total # Renormalize 1st time
    
    # 2. Apply Temperature
    if temperature == 0:
        # Greedy: Argmax
        idx = int(np.argmax(probs))
        return converter.decode(idx)
    else:
        # Avoid numerical instability with very small temperatures (though T=1 is standard)
        # Taking power: p_i^(1/T)
        probs = np.power(probs, 1.0 / temperature)
        probs /= probs.sum() # Renormalize 2nd time
        
    idx = int(np.random.choice(len(probs), p=probs))
    move = converter.decode(idx)
    
    # Safety fallback (shouldn't happen with correct masking)
    if move is None:
        return converter.decode(int(np.argmax(probs)))
        
    return move


import numpy as np
import chess

def board_to_tensor(board: chess.Board):
    """
    Converts a chess.Board object into a 12x8x8 numpy array (float32).
    Shape: (12 channels, 8 rows, 8 columns)
    """
    # 1. Initialize empty matrix
    # Order: P, N, B, R, Q, K (White) then P, N, B, R, Q, K (Black)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # 2. Define mapping from Piece Type to Channel Offset
    # chess.PAWN = 1, KNIGHT = 2, ... KING = 6
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # 3. Iterate over all 64 squares to fill the tensor
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        
        if piece:
            # Determine rank (row) and file (col)
            # Note: chess.SQUARES goes A1, B1... H1, A2...
            # We want matrix coordinates: row 0 is rank 1, row 7 is rank 8
            row = chess.square_rank(square) 
            col = chess.square_file(square)

            # Determine channel
            # White pieces: 0-5, Black pieces: 6-11
            channel = piece_map[piece.piece_type]
            if piece.color == chess.BLACK:
                channel += 6
            
            # Set the bit to 1
            tensor[channel, row, col] = 1.0

    return tensor

import sys
import psutil
import os


# 2. The Memory Checker
def check_memory(logger, step_name):
    """
    Logs current memory usage. 
    step_name: Where are we? (e.g. 'Start of MCTS', 'After Training')
    """
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3) # Convert bytes to GB
    
    logger.debug(f"MEM CHECK [{step_name}]: {mem_gb:.2f} GB")
    
    # Optional: Safety tripwire
    if mem_gb > 16: # Adjust to your RAM limit
        logger.error("!!! MEMORY CRITICAL !!! Shutting down safely.")
        sys.exit(1)
