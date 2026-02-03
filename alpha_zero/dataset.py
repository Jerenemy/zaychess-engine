from __future__ import annotations

from typing import Iterable, Sequence
from collections import deque
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class AlphaZeroDataset(Dataset):
    # dont need a custom collate function since the inputs are the same size 
    # (collate is when have "state": s1, "state": s2, convert to "state": Tensor(s1,s2))
    def __init__(self, entries: Sequence[tuple[np.ndarray, Sequence[float], float]]) -> None:
        self.entries = entries
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # get raw data tuple
        board, policy, value = self.entries[idx]
        
        # assuming state is already a 12x8x8 numpy array, and convert to tensor
        # if its a python chess board object, need to convert here
        state_tensor = torch.from_numpy(board)
        
        # convert policy to tensor: its a prob dist with 4672 vals (all possible moves)
        policy_tensor = torch.tensor(policy, dtype=torch.float32)
        
        # convert value to tensor: it's a single float of either -1.0, 0.0, 1.0
        value_tensor = torch.tensor(value, dtype=torch.float32)
        
        # return a dict to handle multiple outputs
        return {
            "state": state_tensor,
            "policy": policy_tensor,
            "value": value_tensor
        }
    

def label_data(
    game_data: Sequence[tuple[np.ndarray, Sequence[float], int]],
    result_str: str,
) -> list[tuple[np.ndarray, Sequence[float], float]]:
    """
    game_data: list of (board_tensor, policy, turn)
    result_str: "1-0", "0-1", or "1/2-1/2"
    returns: list of (board_tensor, policy, winner_label)
    """
    # 1. Determine the global winner
    global_winner = None
    if result_str == "1-0":
        global_winner = 1 # white
    elif result_str == "0-1":
        global_winner = -1 # black
    # else "1/2-1/2" implies global_winner is None (Draw)

    labeled_data = []
    
    # 2. Assign rewards relative to the player whose turn it was
    for board, policy, turn in game_data:
        
        # Case A: Draw
        if global_winner is None:
            value = 0.0
            
        # Case B: The current player matches the winner
        elif turn == global_winner:
            value = 1.0
            
        # Case C: The current player lost
        else:
            value = -1.0
            
        labeled_data.append((board, policy, value))
        
    return labeled_data



class Buffer:
    def __init__(self, maxlen: int) -> None:
        # maxlen: max num moves to store
        # deque w maxlen automatically handles sliding window logic
        self.memory: deque[tuple[np.ndarray, Sequence[float], float]] = deque(maxlen=maxlen)
        self.maxlen: int = maxlen
        
    def add(self, experience: Sequence[tuple[np.ndarray, Sequence[float], float]] | tuple[np.ndarray, Sequence[float], float]) -> None:
        """
        experience: tuple (state, policy, val) or list of tuples
        """
        if isinstance(experience, list):
            self.memory.extend(experience)
        else:
            self.memory.append(experience)
     
    def sample_batch(self, batch_size: int) -> list[tuple[np.ndarray, Sequence[float], float]]:
        """
        returns random sample of batch_size. if buffer smaller than batch_size, returns entire buffer
        """
        if len(self.memory) < batch_size:
            return list(self.memory)
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)
    
