import os
import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import chess

from mcts import MCTS, Node
from model import AlphaZeroNet
from dataset import ChessDataset, label_data, Buffer
from utils import converter

def save_checkpoint(model, optimizer, gen, ckpt_dir: Path, epoch=None, buffer_len=None):
    """Write a checkpoint to disk and return the checkpoint path as a string."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    name = f"az_gen_{gen}"
    if epoch is not None:
        name += f"_epoch_{epoch}"

    path = ckpt_dir / f"{name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gen": gen,
            "epoch": epoch,
            "buffer_len": buffer_len,
            "num_actions": 4672,
        },
        path,
    )
    return str(path)

def run_train_epoch(model, dataloader, device, optimizer, label="policy"):
    """Run one training epoch over the dataloader for the selected target label."""
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        X = batch["state"].to(device)
        policy_logits_pred, value_pred = model(X)

        if label == "policy":
            y_true = batch["policy"].to(device)
            log_probs = F.log_softmax(policy_logits_pred, dim=1)
            loss = -(y_true * log_probs).sum(dim=1).mean()
        elif label == "value":
            y_true = batch["value"].to(device).unsqueeze(1)
            loss = F.mse_loss(value_pred, y_true)
        else:
            raise ValueError("Invalid label, either policy or value")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))

def parse_args():
    """Parse CLI arguments for training configuration."""
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default="runs/dev")
    p.add_argument("--num-gens", type=int, default=100)
    p.add_argument("--num-epochs", type=int, default=40)
    p.add_argument("--mcts-steps", type=int, default=100)
    p.add_argument("--buffer-maxlen", type=int, default=10000)
    p.add_argument("--buffer-batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    return p.parse_args()

def main():
    """Entry point for self-play training."""
    args = parse_args()

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet((12, 8, 8), num_actions=4672).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    print(f"using device: {device}")
    print(f"run_dir: {run_dir}")

    buffer = Buffer(maxlen=args.buffer_maxlen)
    board = chess.Board()

    for gen in range(args.num_gens):
        node = Node(board.root(), "0000")
        new_data = []

        while not node.is_game_over():
            mcts = MCTS(node, model)
            mcts.run(args.mcts_steps)
            raw_policy = node.get_policy_dict()
            policy_array = converter.policy_to_tensor(raw_policy)
            new_data.append((node, policy_array, None))
            node = node.apply_move_from_dist(policy_array)

        result_str = node.result()
        print(result_str)

        labeled_data = label_data(new_data, result_str)
        buffer.add(labeled_data)

        raw_batch = buffer.sample_batch(args.buffer_batch_size)
        dataset = ChessDataset(raw_batch)
        train_loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
        )

        last_epoch = None
        for epoch in range(args.num_epochs):
            policy_train_loss = run_train_epoch(model, train_loader, device, optimizer, label="policy")
            value_train_loss = run_train_epoch(model, train_loader, device, optimizer, label="value")
            print(f"gen {gen}, epoch {epoch}, prob_train_loss: {policy_train_loss}, val_train_loss: {value_train_loss}")
            last_epoch = epoch

        ckpt_path = save_checkpoint(model, optimizer, gen, ckpt_dir, epoch=last_epoch, buffer_len=len(buffer))
        print(f"saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    main()
