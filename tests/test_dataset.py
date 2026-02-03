import pytest
import chess
import torch
import numpy as np
from alpha_zero.dataset import AlphaZeroDataset, Buffer, label_data

def test_label_data_white_wins():
    # Setup: 2 positions, White turn for both, White eventually wins
    board1 = np.zeros((12, 8, 8), dtype=np.float32)
    board2 = np.zeros((12, 8, 8), dtype=np.float32)
    
    game_data = [
        (board1, [0]*4672, 1),
        (board2, [0]*4672, 1)
    ]
    result_str = "1-0"
    
    labeled = label_data(game_data, result_str)
    
    assert len(labeled) == 2
    # Both were White's turn, White won -> z=1.0 for both
    assert labeled[0][2] == 1.0
    assert labeled[1][2] == 1.0

def test_label_data_black_wins():
    # Setup: 2 positions, White turn then Black turn, Black eventually wins
    board1 = np.zeros((12, 8, 8), dtype=np.float32)
    board2 = np.zeros((12, 8, 8), dtype=np.float32)
    
    game_data = [
        (board1, [0]*4672, 1), # White turn
        (board2, [0]*4672, -1) # Black turn
    ]
    result_str = "0-1"
    
    labeled = label_data(game_data, result_str)
    
    # White turn (board1) and Black won -> White lost -> z=-1.0
    assert labeled[0][2] == -1.0
    # Black turn (board2) and Black won -> Black won -> z=1.0
    assert labeled[1][2] == 1.0

def test_label_data_draw():
    board1 = np.zeros((12, 8, 8), dtype=np.float32)
    game_data = [(board1, [0]*4672, 1)]
    result_str = "1/2-1/2"
    
    labeled = label_data(game_data, result_str)
    assert labeled[0][2] == 0.0

def test_buffer_sampling():
    buffer = Buffer(maxlen=10)
    for i in range(15):
        buffer.add((np.zeros((12,8,8)), [0]*4672, float(i)))
    
    assert len(buffer) == 10
    batch = buffer.sample_batch(5)
    assert len(batch) == 5
    
    batch_large = buffer.sample_batch(20)
    assert len(batch_large) == 10

def test_chess_dataset():
    entries = [
        (np.zeros((12,8,8), dtype=np.float32), np.ones(4672, dtype=np.float32), 1.0)
    ]
    dataset = AlphaZeroDataset(entries)
    assert len(dataset) == 1
    
    item = dataset[0]
    assert isinstance(item["state"], torch.Tensor)
    assert isinstance(item["policy"], torch.Tensor)
    assert isinstance(item["value"], torch.Tensor)
    assert item["state"].shape == (12, 8, 8)
    assert item["policy"].shape == (4672,)
    assert item["value"].item() == 1.0
