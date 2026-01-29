import logging

from .model import AlphaZeroNet
from .mcts import MCTS, Node
from . import utils
from .utils import board_to_tensor, set_game_mode
from .config import Config
from .dataset import label_data
from . import tictactoe as ttt
from . import chess_wrapper as cw
from . import logger_config
logger = logger_config.setup_logger('self_play', level=logging.DEBUG)

def play_one_game(model: AlphaZeroNet, cfg: Config, game_mode='chess'):
    set_game_mode(game_mode)
    if game_mode == 'chess':
        board = cw.Board() 
    elif game_mode == 'tictactoe':
        board = ttt.Board()
    node = Node(board, "0000")
    moves = []
    while not node.is_game_over() and len(moves) < cfg.max_num_moves_per_game:
        mcts = MCTS(node, model)
        mcts.run(cfg.mcts_steps)
        # 1. Get raw visit counts from MCTS (format: {'e2e4': 100, 'g1f3': 50})
        raw_policy = node.get_policy_dict()
        # 2. convert to tensors (policy: to the len-4672 array, format: [0.0, ..., 0.33])
        policy_array = utils.converter.policy_to_tensor(raw_policy)
        # pass only board converted to tensor
        board_tensor = board_to_tensor(node.state)
        turn = 1 if node.state.turn == cw.WHITE else -1
        # 3. Determine temperature (T=1 for first N moves, T=0 otherwise)
        if len(moves) < cfg.temperature_move_threshold:
            temp = 1.0
        else:
            temp = 0.0

        # 4. store and later add to buffer
        moves.append((board_tensor, policy_array, turn, None))
        node = node.apply_move_from_dist(policy_array, temperature=temp)
    result_str = node.result()
    logger.debug(f"Game result: {result_str} ({len(moves)} moves)")
    labeled_data = label_data(moves, result_str)
    return labeled_data, result_str, len(moves)