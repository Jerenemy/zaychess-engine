import pytest
import chess
import numpy as np
from alpha_chess.utils import ActionConverter, board_to_tensor, converter

def test_action_converter_queen_moves():
    # Test a simple vertical move e2e4
    idx = converter.encode("e2e4")
    assert idx is not None
    assert 0 <= idx < 4672
    assert converter.decode(idx) == "e2e4"

    # Test a queen promotion e7e8q
    idx_q = converter.encode("e7e8q")
    assert idx_q is not None
    assert converter.decode(idx_q) == "e7e8q"

def test_action_converter_knight_moves():
    # Test a knight jump g1f3
    idx = converter.encode("g1f3")
    assert idx is not None
    assert converter.decode(idx) == "g1f3"

def test_action_converter_underpromotions():
    # Test a knight promotion e7e8n
    idx_n = converter.encode("e7e8n")
    assert idx_n is not None
    assert converter.decode(idx_n) == "e7e8n"

def test_board_to_tensor_initial():
    board = chess.Board()
    tensor = board_to_tensor(board)
    assert tensor.shape == (12, 8, 8)
    
    # White pawn at A2 (rank 1, file 0)
    # White pawn channel is 0
    # Rank 1 is row 1
    assert tensor[0, 1, 0] == 1.0
    
    # Black king at E8 (rank 7, file 4)
    # Black king channel is 6 + 5 = 11
    # Rank 7 is row 7
    assert tensor[11, 7, 4] == 1.0

def test_board_to_tensor_empty():
    board = chess.Board(None) # Empty board
    tensor = board_to_tensor(board)
    assert np.sum(tensor) == 0.0

def test_policy_to_tensor():
    policy_dict = {"e2e4": 10, "d2d4": 10}
    tensor = converter.policy_to_tensor(policy_dict)
    assert tensor.shape == (4672,)
    assert np.isclose(np.sum(tensor), 1.0)
    
    idx_e2e4 = converter.encode("e2e4")
    idx_d2d4 = converter.encode("d2d4")
    assert tensor[idx_e2e4] == 0.5
    assert tensor[idx_d2d4] == 0.5
