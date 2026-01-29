# import chess
import torch
import random
import logging
from abc import ABC, abstractmethod

from alpha_zero import Config, AlphaZeroNet, MCTS, Node, setup_logger, set_game_mode
from alpha_zero import chess_wrapper as cw
from alpha_zero import tictactoe as ttt

cfg = Config()
logger = setup_logger('evaluate', level=logging.INFO)

class Arena:
    def __init__(self, challenger, baseline):
        self.challenger = challenger
        self.baseline = baseline
    
    def play_game(self, move_first=True, num_moves=20, game_mode='chess'):
        game_lib = cw if game_mode == 'chess' else ttt
        board = game_lib.Board()
        # Challenger is White if move_first is True, else Black
        challenger_color = game_lib.WHITE if move_first else game_lib.BLACK
        
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

    def evaluate(self, num_games=10, game_mode='chess'):
        results_keys = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
        logger.info(f"Evaluating {num_games} games")
        challenger_win_percentage = 0
        for i in range(num_games):
            # Alternate who goes first
            move_first = i % 2 == 0
            result = self.play_game(move_first, game_mode=game_mode)
            result = self._parse_result(result, move_first)
            results_keys[result] = results_keys.get(result, 0) + 1
            
            # Log individual game result clearly
            winner_str = "Draw"
            if result == "1-0":
                winner_str = "Challenger (Model) Won"
                challenger_win_percentage += 1
            elif result == "0-1":
                winner_str = "Baseline (Random) Won"
                
            logger.info(f"Game {i+1}/{num_games} finished: {result} -> {winner_str}")
            
        logger.info("\nFinal Summary:")
        logger.info(f"Challenger Wins: {results_keys['1-0']}")
        logger.info(f"Baseline Wins:   {results_keys['0-1']}")
        logger.info(f"Draws:           {results_keys['1/2-1/2']}")
        
        return challenger_win_percentage / num_games

class Player(ABC):
    def __init__(self, model, device, game_mode='chess'):
        self.model = model
        self.device = device
        self.game_lib = cw if game_mode == 'chess' else ttt
    
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
        best_move = max(policy_dict, key=policy_dict.get)
        return best_move
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load("checkpoints/az_gen_9_epoch_0.pt", map_location=device)
    game_mode = 'tictactoe' # 'chess' or 'tictactoe'
    set_game_mode(game_mode)
    if game_mode == 'tictactoe':
        model = AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9, num_resblocks=5)
    else:
        model = AlphaZeroNet()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from generation {checkpoint.get('gen', 'unknown')}")
    arena = Arena(MCTSPlayer(model, device, game_mode=game_mode), RandomPlayer(None, device, game_mode=game_mode))
    results = arena.evaluate(num_games=4, game_mode=game_mode)
    print("\nFinal results:")
    print(results)

if __name__ == "__main__":
    main()

