import pytest
import numpy as np
import chess
from alpha_zero.utils import sample_next_move, converter

def test_sample_next_move_temperature_1():
    # Deterministic fallback check for single move
    legal_moves = [chess.Move.from_uci("e2e4")]
    # policy for e2e4 = 1.0, others 0.0
    move_probs = np.zeros(4672)
    idx = converter.encode("e2e4")
    move_probs[idx] = 1.0
    
    move = sample_next_move(move_probs, legal_moves, temperature=1.0)
    assert move == "e2e4"

def test_sample_next_move_temperature_0_greedy():
    # Two moves, one has slightly higher prob
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    move_probs = np.zeros(4672)
    idx_e4 = converter.encode("e2e4")
    idx_d4 = converter.encode("d2d4")
    
    move_probs[idx_e4] = 0.6
    move_probs[idx_d4] = 0.4
    
    # T=0 should ALWAYS pick the max
    for _ in range(10):
        move = sample_next_move(move_probs, legal_moves, temperature=0.0)
        assert move == "e2e4"

def test_sample_next_move_temperature_high():
    # High temp -> uniform distribution
    legal_moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("d2d4")]
    move_probs = np.zeros(4672)
    idx_e4 = converter.encode("e2e4")
    idx_d4 = converter.encode("d2d4")
    
    move_probs[idx_e4] = 0.9
    move_probs[idx_d4] = 0.1
    
    # With T=100, probs become very close to 0.5/0.5
    # Just checking it doesn't crash and returns valid move
    move = sample_next_move(move_probs, legal_moves, temperature=100.0)
    assert move in ["e2e4", "d2d4"]
