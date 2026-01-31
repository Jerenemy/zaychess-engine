
import unittest
from unittest.mock import MagicMock, patch
import sys
import io
import chess
from uci_engine import UCIEngine
from alpha_zero.chess_wrapper import Board
from alpha_zero import Node

class TestUCIEngine(unittest.TestCase):
    def setUp(self):
        # Patch init to avoid loading model/device
        with patch('uci_engine.AlphaZeroNet'), \
             patch('torch.load'), \
             patch('uci_engine.set_game_mode'), \
             patch('uci_engine.logger'):
            self.engine = UCIEngine("dummy_ckpt", 10, False)
            # Mock the model behavior
            self.engine.model.predict_value = MagicMock(return_value=(
                [0.1]*4672, 0.0 # uniform policy, 0 value
            ))

    def test_uci_handshake(self):
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            self.engine.handle_command("uci")
            output = mock_stdout.getvalue()
            self.assertIn("id name ZayChess AlphaZero", output)
            self.assertIn("uciok", output)

    def test_isready(self):
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            self.engine.handle_command("isready")
            output = mock_stdout.getvalue()
            self.assertIn("readyok", output)

    def test_ucinewgame(self):
        # Make some moves first
        self.engine.board.push("e2e4")
        self.assertNotEqual(len(self.engine.board.move_stack), 0)
        
        self.engine.handle_command("ucinewgame")
        self.assertEqual(len(self.engine.board.move_stack), 0)
        self.assertIsInstance(self.engine.board, Board)

    def test_position_startpos(self):
        self.engine.handle_command("position startpos moves e2e4 e7e5")
        self.assertEqual(len(self.engine.board.move_stack), 2)
        self.assertEqual(self.engine.board.board.piece_at(chess.E4).symbol(), 'P')

    def test_position_fen(self):
        # Ruy Lopez FEN
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
        self.engine.handle_command(f"position fen {fen}")
        self.assertEqual(self.engine.board.board.fen(), fen)

    def test_position_fen_moves(self):
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        self.engine.handle_command(f"position fen {fen} moves e7e5")
        self.assertEqual(len(self.engine.board.move_stack), 1)

    def test_go(self):
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # Setup a board
            self.engine.handle_command("position startpos")
            # Run go
            self.engine.handle_command("go")
            
            output = mock_stdout.getvalue()
            self.assertIn("bestmove", output)
