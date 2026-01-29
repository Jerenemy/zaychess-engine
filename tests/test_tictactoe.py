import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import unittest
from alpha_zero.self_play import play_one_game
from alpha_zero.model import AlphaZeroNet
from alpha_zero.config import Config
from alpha_zero.utils import set_game_mode

class TestTicTacToe(unittest.TestCase):
    def test_play_one_game_ttt(self):
        # Setup
        # TTT: 2 channels, 3x3 board, 9 actions
        model = AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9, resblock_dim=16, num_resblocks=2)
        cfg = Config(mcts_steps=10, max_num_moves_per_game=20, temperature_move_threshold=5)
        device = torch.device('cpu')
        
        # Run
        print("Starting TTT game...")
        labeled_data, result, num_moves = play_one_game(model, cfg, device, game_mode='tictactoe')
        
        print(f"Game finished. Result: {result}, Moves: {num_moves}")
        
        # Verify
        self.assertIn(result, ["1-0", "0-1", "1/2-1/2"])
        self.assertTrue(len(labeled_data) > 0)
        self.assertEqual(len(labeled_data), num_moves)
        
        # Check data shapes
        state_tensor, policy, val = labeled_data[0]
        self.assertEqual(state_tensor.shape, (2, 3, 3))
        self.assertEqual(len(policy), 9)

if __name__ == '__main__':
    unittest.main()
