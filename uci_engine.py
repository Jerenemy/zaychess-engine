import sys
import argparse
import logging
import chess
import torch
from alpha_chess import AlphaZeroNet, MCTS, Node, Config

# Setup logging
logging.basicConfig(
    filename='uci_engine.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('uci')

def parse_args():
    parser = argparse.ArgumentParser(description='AlphaZero UCI Engine')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--mcts-steps', type=int, default=100, help='Number of MCTS simulations per move')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    return parser.parse_args()

class UCIEngine:
    def __init__(self, checkpoint_path, mcts_steps, use_cuda):
        self.mcts_steps = mcts_steps
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load Model
        self.model = AlphaZeroNet()
        try:
            # Load full checkpoint dict, then extract 'model' key
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            else:
                # Fallback if it's just the state dict (legacy/manual saves)
                self.model.load_state_dict(checkpoint)
                
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            sys.exit(1)
            
        self.board = chess.Board()

    def loop(self):
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                logger.debug(f"UCI < {line}")
                self.handle_command(line)
            except Exception as e:
                logger.error(f"Error in loop: {e}")
                
    def handle_command(self, cmd_line):
        tokens = cmd_line.split()
        if not tokens:
            return
        
        cmd = tokens[0]
        
        if cmd == "uci":
            print("id name ZayChess AlphaZero")
            print("id author Jerenemy")
            print("uciok")
            sys.stdout.flush()
            
        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()
            
        elif cmd == "ucinewgame":
            self.board = chess.Board()
            
        elif cmd == "position":
            self._handle_position(tokens)
            
        elif cmd == "go":
            self._handle_go(tokens)
            
        elif cmd == "quit":
            sys.exit(0)

    def _handle_position(self, tokens):
        # position [startpos | fen <fen>] [moves <moves>]
        idx = 1
        if tokens[idx] == "startpos":
            self.board = chess.Board()
            idx += 1
        elif tokens[idx] == "fen":
            # Fen is usually 6 tokens long? Join until 'moves' or end
            idx += 1
            fen_tokens = []
            while idx < len(tokens) and tokens[idx] != "moves":
                fen_tokens.append(tokens[idx])
                idx += 1
            fen = " ".join(fen_tokens)
            self.board = chess.Board(fen)
            
        if idx < len(tokens) and tokens[idx] == "moves":
            idx += 1
            for move_uci in tokens[idx:]:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in self.board.legal_moves:
                        self.board.push(move)
                    else:
                        logger.error(f"Illegal move received: {move_uci}")
                except ValueError:
                    logger.error(f"Invalid move format: {move_uci}")

    def _handle_go(self, tokens):
        # We ignore time controls for now and just use fixed nodes
        # In a real engine, we'd parse wtime/btime/movetime
        
        # Create MCTS root from current board
        # Dummy move "0000" for root
        root = Node(self.board, "0000")
        mcts = MCTS(root, self.model)
        
        # Run search
        mcts.run(self.mcts_steps)
        
        # Get best move (most visited)
        policy = root.get_policy_dict()
        if not policy:
            # Fallback if no moves? (Game over or 0 steps)
            if self.board.legal_moves:
                best_move = list(self.board.legal_moves)[0]
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove (none)")
            sys.stdout.flush()
            return

        best_move_uci = max(policy, key=policy.get)
        print(f"bestmove {best_move_uci}")
        sys.stdout.flush()
        logger.info(f"Best move: {best_move_uci}, Visits: {policy[best_move_uci]}")

if __name__ == "__main__":
    args = parse_args()
    engine = UCIEngine(args.checkpoint, args.mcts_steps, args.cuda)
    engine.loop()
