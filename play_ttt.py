import torch
from alpha_zero import Config, AlphaZeroNet, MCTS, Node
from alpha_zero.game_adapter import TicTacToeAdapter
from alpha_zero import tictactoe as ttt

# Reuse MCTSPlayer from evaluate.py logic or redefine it here since evaluate is not a package
# To avoid duplication, I'll redefine a simple version or import if evaluate was refactored.
# evaluate.py is a script, not easily importable without running main. 
# I will copy the MCTSPlayer logic here for simplicity.

class MCTSPlayer:
    def __init__(self, model, device, mcts_steps):
        self.model = model
        self.device = device
        self.adapter = TicTacToeAdapter()
        self.mcts_steps = mcts_steps
    
    def get_move(self, board):
        node = Node(board, "0000")
        mcts = MCTS(node, self.model, self.adapter)
        # Use more steps for better play
        mcts.run(self.mcts_steps * 4) 
        
        policy_dict = node.get_policy_dict(normalized=True)
        
        # Sort and Display Probabilities
        sorted_moves = sorted(policy_dict.items(), key=lambda x: x[1], reverse=True)
        print("\nAI Move Probabilities:")
        print(f"{'Move':<6} | {'Prob':<6}")
        print("-" * 15)
        for move, prob in sorted_moves:
             print(f"{move:<6} | {prob:.4f}")
        print("-" * 15)

        best_move = sorted_moves[0][0] # Pick the explicit max
        return best_move

class HumanPlayer:
    def get_move(self, board):
        legal_moves = [str(m) for m in board.legal_moves]
        print(f"Legal moves: {legal_moves}")
        while True:
            inp = input("Enter your move (0-8): ").strip()
            if inp in legal_moves:
                # We need to return the same type as MCTSPlayer returns (generic move or string?)
                # MCTSPlayer returns the key from policy_dict which is associated_move (string)
                # But ttt.Board.push accepts string or Move object
                # Let's return the Move object matching the input or just the string?
                # ttt.Board.legal_moves returns Move objects. str(m) gives the uci string (int).
                return inp
            print("Invalid move, try again.")

def main():
    cfg = Config()
    game_mode = 'tictactoe'
    # elif game_mode == 'chess':
    #     adapter = ChessAdapter()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Init Model
    model = AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9, num_resblocks=5)
    
    # Load Checkpoint - try to find latest or specific one
    # For now hardcoding the one we know exists or handling error
    ckpt_path = "checkpoints/az_gen_19_epoch_1.pt"
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"Loaded model from {ckpt_path}")
    except FileNotFoundError:
        print(f"Checkpoint {ckpt_path} not found. Playing with random weights!")
    
    model.to(device)
    model.eval()

    ai_player = MCTSPlayer(model, device, cfg.mcts_steps)
    human_player = HumanPlayer()

    # Game Loop
    while True:
        board = ttt.Board()
        # Human is 'X' (1, White/True), AI is 'O' (-1, Black/False)
        # Or let user choose? Let's default Human goes first.
        
        human_turn = True
        
        print("\n=== New Game ===")
        print("You are X (First). AI is O.")
        
        while not board.is_game_over():
            print(f"\n{board}")
            
            if human_turn:
                print("Your turn (X):")
                move_uci = human_player.get_move(board)
            else:
                print("AI Thinking...")
                move_uci = ai_player.get_move(board)
                print(f"AI chose: {move_uci}")

            board.push_uci(move_uci)
            human_turn = not human_turn
            
        print(f"\n{board}")
        print(f"Game Over! Result: {board.result()}")
        
        play_again = input("Play again? (y/n): ").lower()
        if play_again != 'y':
            break

if __name__ == '__main__':
    main()
