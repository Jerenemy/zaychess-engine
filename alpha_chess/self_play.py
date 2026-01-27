import chess
from torch import device

from .model import AlphaZeroNet
from .mcts import MCTS, Node
from .utils import converter, board_to_tensor
from .config import Config
from .dataset import label_data

def play_one_game(model: AlphaZeroNet, cfg: Config, device: device):
    board = chess.Board()
    node = Node(board, "0000")
    new_data = []
    while not node.is_game_over():
        mcts = MCTS(node, model)
        mcts.run(cfg.mcts_steps)
        # 1. Get raw visit counts from MCTS (format: {'e2e4': 100, 'g1f3': 50})
        raw_policy = node.get_policy_dict()
        # 2. convert to tensors (policy: to the len-4672 array, format: [0.0, ..., 0.33])
        policy_array = converter.policy_to_tensor(raw_policy)
        # pass only board converted to tensor
        board_tensor = board_to_tensor(node.state)
        turn = 1 if node.state.turn == chess.WHITE else -1
        # 3. store and later add to buffer
        new_data.append((board_tensor, policy_array, turn, None))
        node = node.apply_move_from_dist(policy_array)
    result_str = node.result()
    # logger.info(f"Game result: {result_str} ({len(new_data)} moves)")
    labeled_data = label_data(new_data, result_str)
    return labeled_data, result_str, len(new_data)