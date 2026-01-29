import unittest
from alpha_zero.chess_wrapper import Board, Move, WHITE, BLACK

class TestChessWrapper(unittest.TestCase):
    def test_board_init(self):
        board = Board()
        self.assertEqual(board.turn, WHITE)
        self.assertEqual(len(board.legal_moves), 20)

    def test_push_pop(self):
        board = Board()
        move = Move("e2e4")
        board.push(move)
        
        self.assertEqual(board.turn, BLACK)
        self.assertEqual(len(board.move_stack), 1)
        # pop
        board.pop()
        self.assertEqual(board.turn, WHITE)

    def test_push_uci(self):
        board = Board()
        board.push_uci("e2e4")
        self.assertEqual(board.turn, BLACK)

    def test_legal_moves(self):
        board = Board()
        moves = board.legal_moves
        self.assertTrue(all(isinstance(m, Move) for m in moves))
        self.assertEqual(len(moves), 20)

    def test_result(self):
        board = Board()
        self.assertEqual(board.result(), "*")

if __name__ == '__main__':
    unittest.main()
