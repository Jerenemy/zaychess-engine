import logging

from .model import AlphaZeroNet
from .mcts import MCTS, Node
from .utils import sample_next_move
from .config import Config
from .dataset import label_data
from . import logger_config
from .game_adapter import GameAdapter
logger = logger_config.setup_logger('self_play', level=logging.DEBUG)

def play_one_game(model: AlphaZeroNet, cfg: Config, adapter: GameAdapter):
    board = adapter.new_board()
    # if game_mode == 'chess':
    #     board = cw.Board() 
    # elif game_mode == 'tictactoe':
    #     board = ttt.Board()
    node = Node(board, "0000")
    moves = []
    while not adapter.is_terminal(node.state) and len(moves) < cfg.max_num_moves_per_game:
        mcts = MCTS(node, model, adapter)
        mcts.run(cfg.mcts_steps)
        # 1. Get raw visit counts from MCTS (format: {'e2e4': 100, 'g1f3': 50})
        raw_policy = node.get_policy_dict()
        # 2. convert to tensors (policy: to the len-4672 array (for chess), format: [0.0, ..., 0.33])
        policy_array = adapter.policy_to_tensor(raw_policy)
        # pass only board converted to tensor
        board_tensor = adapter.board_to_tensor(node.state)
        turn = 1 if getattr(node.state, "turn", True) in (True, 1) else -1
        # 3. Determine temperature (T=1 for first N moves, T=0 otherwise)
        if len(moves) < cfg.temperature_move_threshold:
            temp = 1.0
        else:
            temp = 0.0

        # 4. store and later add to buffer
        moves.append((board_tensor, policy_array, turn, None))
        legal_moves = list(adapter.legal_moves(node.state))
        next_move_uci = sample_next_move(
            policy_array,
            legal_moves,
            temperature=temp,
            encode_move=adapter.encode_move,
            decode_move=adapter.decode_move,
        )
        next_state = adapter.copy_board(node.state)
        adapter.push(next_state, next_move_uci)
        node = Node(next_state, next_move_uci)
    result_str = adapter.result(node.state)
    logger.debug(f"Game result: {result_str} ({len(moves)} moves)")
    labeled_data = label_data(moves, result_str)
    return labeled_data, result_str, len(moves)
