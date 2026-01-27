import chess
import torch
import random
import logging
from abc import ABC, abstractmethod

from alpha_chess import Config, AlphaZeroNet, MCTS, Node, setup_logger


cfg = Config()
logger = setup_logger('evaluate', level=logging.INFO)

class Arena:
    def __init__(self, challenger, baseline):
        self.challenger = challenger
        self.baseline = baseline
    
    def play_game(self, move_first=True, num_moves=200):
        board = chess.Board()
        # Challenger is White if move_first is True, else Black
        challenger_color = chess.WHITE if move_first else chess.BLACK
        
        while not board.is_game_over():
            if len(board.move_stack) >= num_moves:
                return "*"
            if board.turn == challenger_color:
                move = self.challenger.get_move(board)
            else:
                move = self.baseline.get_move(board)
            board.push(move)
        # board.result() returns the winner with respect to the player who moved first
        return board.result()
    
    def _parse_result(self, result, move_first):
        if move_first:
            return result
        else:
            return "1-0" if result == "0-1" else "0-1" if result == "1-0" else result

    def evaluate(self, num_games=10):
        results_keys = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        logger.info(f"Evaluating {num_games} games")
        challenger_win_percentage = 0
        for i in range(num_games):
            # Alternate who goes first
            move_first = i % 2 == 0
            result = self.play_game(move_first)
            result = self._parse_result(result, move_first)
            results_keys[result] = results_keys.get(result, 0) + 1
            if result == "1-0":
                challenger_win_percentage += 1
            logger.info(f"Game {i+1}/{num_games} finished: {result}")
        logger.info("\nFinal results:")
        logger.info(results_keys)
        return challenger_win_percentage / num_games

class Player(ABC):
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    @abstractmethod
    def get_move(self, board):
        pass

class RandomPlayer(Player):
    def get_move(self, board):
        return random.choice(list(board.legal_moves))

class MCTSPlayer(Player):
    def get_move(self, board):
        # 1. Create a Node from the current board
        # "0000" is a dummy move for the root
        node = Node(board, "0000")
        
        # 2. Run MCTS
        mcts = MCTS(node, self.model)
        mcts.run(cfg.mcts_steps)
        
        # 3. Pick the move with the most visits
        policy_dict = node.get_policy_dict()
        best_move_uci = max(policy_dict, key=policy_dict.get)
        return chess.Move.from_uci(best_move_uci)
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AlphaZeroNet()
    # FIX: Extract the model weights from the checkpoint dictionary
    checkpoint = torch.load("checkpoints/az_gen_2_epoch_0.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from generation {checkpoint.get('gen', 'unknown')}")

    arena = Arena(MCTSPlayer(model, device), RandomPlayer(None, device))
    results = arena.evaluate(num_games=4)
    print("\nFinal results:")
    print(results)

if __name__ == "__main__":
    main()

