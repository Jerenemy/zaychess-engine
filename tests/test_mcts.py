import pytest
import chess
from alpha_chess.mcts import Node, MCTS
from alpha_chess.model import AlphaZeroNet

def test_node_expansion():
    board = chess.Board()
    node = Node(board, "0000")
    assert node.is_leaf()
    node.expand_children()
    assert not node.is_leaf()
    # Initial position has 20 legal moves
    assert len(node.children) == 20
    
    # Check that children have correct associated moves
    legal_moves = [move.uci() for move in board.legal_moves]
    for child in node.children:
        assert child.associated_move in legal_moves

def test_node_terminal_value():
    # Scholar's mate position (White wins)
    board = chess.Board("r1bqkbnr/ppppp2p/n7/5ppQ/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 1 3")
    assert board.is_game_over()
    
    # It's currently Black's turn, but game is over.
    # result() should be "1-0"
    node = Node(board, "h5g6") # Dummy incoming move
    
    # If it's Black's turn and White won, Black should get -1.0
    val = node.get_terminal_value()
    assert val == -1.0

def test_node_terminal_value_draw():
    # Stalemate position
    board = chess.Board("k7/8/K7/8/8/8/8/8 b - - 0 1") 
    # Not actually stalemate, let's use a clear one
    board = chess.Board("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1")
    assert board.is_stalemate()
    
    node = Node(board, "f7f8")
    assert node.get_terminal_value() == 0.0
