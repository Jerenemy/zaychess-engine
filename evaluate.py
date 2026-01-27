import chess
import torch
import random
from abc import ABC, abstractmethod
from alpha_chess import Config, AlphaZeroNet, MCTS, Node
from torch import load

cfg = Config()

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
            
        return board.result()
    
    def evaluate(self, num_games=10):
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        for i in range(num_games):
            # Alternate who goes first
            result = self.play_game(move_first=(i % 2 == 0))
            results[result] = results.get(result, 0) + 1
            print(f"Game {i+1}/{num_games} finished: {result}")
        return results

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
    checkpoint = load("checkpoints/az_gen_2_epoch_0.pt", map_location=device)
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

