import argparse
import sys
import random

import chess
import torch

from mcts import MCTS, Node
from model import AlphaZeroNet

ENGINE_NAME = "ZayChessEngine"
ENGINE_AUTHOR = "Jeremy"


def send(msg: str) -> None:
    """Write a single UCI response line to stdout."""
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def load_model(checkpoint_path: str | None, device: torch.device) -> AlphaZeroNet:
    """Load an AlphaZeroNet from an optional checkpoint onto the target device."""
    model = AlphaZeroNet((12, 8, 8), num_actions=4672)
    model.to(device)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print("warning: no checkpoint provided; using random weights", file=sys.stderr)
    model.eval()
    return model


class UCIEngine:
    def __init__(self, model: AlphaZeroNet, mcts_steps: int = 200):
        """Initialize the engine state for a UCI session."""
        self.model = model
        self.mcts_steps = max(1, mcts_steps)
        self.board = chess.Board()

    def handle_uci(self) -> None:
        """Handle the UCI identification handshake."""
        send(f"id name {ENGINE_NAME}")
        send(f"id author {ENGINE_AUTHOR}")
        send("option name MCTS_Steps type spin default 200 min 1 max 10000")
        send("uciok")

    def handle_isready(self) -> None:
        """Respond to UCI readiness checks."""
        send("readyok")

    def handle_ucinewgame(self) -> None:
        """Reset the board for a new game."""
        self.board = chess.Board()

    def handle_setoption(self, tokens: list[str]) -> None:
        """Apply UCI option changes from tokenized input."""
        if "name" not in tokens:
            return
        name_idx = tokens.index("name") + 1
        if "value" in tokens:
            value_idx = tokens.index("value")
            name = " ".join(tokens[name_idx:value_idx]).strip().lower()
            value = " ".join(tokens[value_idx + 1 :]).strip()
        else:
            name = " ".join(tokens[name_idx:]).strip().lower()
            value = ""
        if name == "mcts_steps":
            try:
                self.mcts_steps = max(1, int(value))
            except ValueError:
                pass

    def handle_position(self, tokens: list[str]) -> None:
        """Set up the board from a UCI position command."""
        if not tokens:
            return
        if tokens[0] == "startpos":
            board = chess.Board()
            move_tokens = tokens[1:]
        elif tokens[0] == "fen":
            if "moves" in tokens:
                moves_idx = tokens.index("moves")
                fen_tokens = tokens[1:moves_idx]
                move_tokens = tokens[moves_idx + 1 :]
            else:
                fen_tokens = tokens[1:]
                move_tokens = []
            fen = " ".join(fen_tokens).strip()
            try:
                board = chess.Board(fen)
            except ValueError:
                board = chess.Board()
                print("warning: invalid FEN, resetting board", file=sys.stderr)
        else:
            return

        if move_tokens and move_tokens[0] == "moves":
            move_tokens = move_tokens[1:]
        for move_uci in move_tokens:
            try:
                board.push_uci(move_uci)
            except ValueError:
                print(f"warning: invalid move {move_uci}", file=sys.stderr)
                break
        self.board = board

    def select_best_move(self, board: chess.Board, steps: int) -> str | None:
        """Run MCTS on the given board and return a chosen move."""
        if board.is_game_over():
            return None
        root = Node(board.copy(), "0000")
        mcts = MCTS(root, self.model)
        mcts.run(max(1, steps))
        if not root.children:
            root.expand_children()
        if not root.children:
            return None
        max_visits = max(child.visits for child in root.children)
        if max_visits > 0:
            candidates = [c for c in root.children if c.visits == max_visits]
        else:
            max_prior = max(child.prior for child in root.children)
            candidates = [c for c in root.children if c.prior == max_prior]
        return random.choice(candidates).associated_move

    def handle_go(self, tokens: list[str]) -> None:
        """Handle a UCI go command and emit the chosen move."""
        steps = self.mcts_steps
        if "nodes" in tokens:
            try:
                steps = max(1, int(tokens[tokens.index("nodes") + 1]))
            except (ValueError, IndexError):
                pass

        best_move = self.select_best_move(self.board, steps)
        if best_move is None:
            send("bestmove 0000")
            return
        try:
            self.board.push_uci(best_move)
        except ValueError:
            pass
        send(f"bestmove {best_move}")


def main() -> None:
    """Parse CLI args and run the UCI event loop."""
    parser = argparse.ArgumentParser(description="UCI interface for ZayChessEngine")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--checkpoint", default=None, help="path to .pt checkpoint")
    parser.add_argument("--mcts-steps", type=int, default=200)
    parser.add_argument("--device", default=default_device, choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)
    engine = UCIEngine(model, mcts_steps=args.mcts_steps)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        cmd = tokens[0]
        if cmd == "uci":
            engine.handle_uci()
        elif cmd == "isready":
            engine.handle_isready()
        elif cmd == "ucinewgame":
            engine.handle_ucinewgame()
        elif cmd == "setoption":
            engine.handle_setoption(tokens[1:])
        elif cmd == "position":
            engine.handle_position(tokens[1:])
        elif cmd == "go":
            engine.handle_go(tokens[1:])
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
