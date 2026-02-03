import chess
from unittest.mock import MagicMock

from alpha_zero.mcts import Node, MCTS
from alpha_zero.game_adapter import ChessAdapter
from alpha_zero import chess_wrapper as cw

def test_node_expansion():
    adapter = ChessAdapter()
    board = adapter.new_board()
    node = Node(board, "0000")
    mcts = MCTS(node, MagicMock(), adapter)
    assert node.is_leaf()
    mcts.expand_children(node)
    assert not node.is_leaf()
    # Initial position has 20 legal moves
    assert len(node.children) == 20
    
    # Check that children have correct associated moves
    legal_moves = [move.uci() for move in adapter.legal_moves(board)]
    for child in node.children:
        assert child.associated_move in legal_moves

def test_node_terminal_value():
    # Scholar's mate position (White wins)
    adapter = ChessAdapter()
    board = cw.Board(chess.Board("r1bqkbnr/ppppp2p/n7/5ppQ/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 1 3"))
    assert board.is_game_over()
    
    # It's currently Black's turn, but game is over.
    # result() should be "1-0"
    node = Node(board, "h5g6") # Dummy incoming move
    mcts = MCTS(node, MagicMock(), adapter)
    
    # If it's Black's turn and White won, Black should get -1.0
    val = mcts.get_terminal_value(node)
    assert val == -1.0

def test_node_terminal_value_draw():
    # Stalemate position
    adapter = ChessAdapter()
    board = cw.Board(chess.Board("k7/8/K7/8/8/8/8/8 b - - 0 1")) 
    # Not actually stalemate, let's use a clear one
    board = cw.Board(chess.Board("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1"))
    assert board.board.is_stalemate()
    
    node = Node(board, "f7f8")
    mcts = MCTS(node, MagicMock(), adapter)
    assert mcts.get_terminal_value(node) == 0.0
