from __future__ import annotations

from typing import Optional
import os
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alpha_zero import (
    Config, 
    AlphaZeroNet, 
    Buffer, 
    AlphaZeroDataset, 
    check_memory,
    play_one_game,
    ChessAdapter
)
from alpha_zero.logger_config import setup_logger

def save_checkpoint(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    gen: int,
    checkpoint_dir: str,
    epoch: Optional[int] = None,
    buffer_len: Optional[int] = None,
    num_actions: int = 4672,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    name = f"az_gen_{gen}"
    if epoch is not None:
        name += f"_epoch_{epoch}"
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "gen": gen,
            "epoch": epoch,
            "buffer_len": buffer_len,
            "num_actions": num_actions,
        },
        path,
    )
    return path

def run_train_epoch(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    label: str = "policy",
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        X = batch["state"].to(device)
        policy_logits_pred, value_pred = model(X)
        if label == "policy":
            y_true = batch["policy"].to(device)
            log_probs = F.log_softmax(policy_logits_pred, dim=1)
            loss = -torch.sum(y_true * log_probs, dim=1).mean()
        elif label == "value":
            y_true = batch["value"].to(device).unsqueeze(1)
            loss = F.mse_loss(value_pred, y_true)
        else: 
            raise ValueError("Invalid label, either policy or value")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main() -> None:
    cfg = Config()
    logger = setup_logger('train', level=logging.DEBUG)
    buffer = Buffer(maxlen=cfg.max_buffer_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlphaZeroNet()
    model.to(device)
    logger.info(f"using device: {device}")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    logger.info(f"num_gens: {cfg.num_gens}, num_epochs: {cfg.num_epochs}, mcts_steps: {cfg.mcts_steps}, buffer_size: {buffer.maxlen}, buffer_batch_size: {cfg.buffer_batch_size}\n")

    for gen in range(cfg.num_gens):
        logger.info(f"Starting generation {gen}")
        check_memory(logger, "Start of Generation") 
        logger.info(f"Buffer length: {len(buffer)}")
        new_gen_data = []
        ### play games ###
        for game in range(cfg.num_games_per_gen):
            new_game_data, result_str, num_moves = play_one_game(
                model,
                cfg,
                adapter=ChessAdapter(),
                logger=logger,
            )
            logger.info(f"Game {game} | Game result: {result_str} ({num_moves} moves)")
            new_gen_data.extend(new_game_data)
            check_memory(logger, f"After game {game}") 
        ### end games ###
        buffer.add(new_gen_data)
        ### train ###
        raw_batch = buffer.sample_batch(cfg.buffer_batch_size) # list of raw tuples: (s1, p1, v1), (s2, p2, v2), ...
        dataset = AlphaZeroDataset(raw_batch)
        train_loader = DataLoader(
            dataset, 
            batch_size=cfg.batch_size, #mini batch size for gpu
            shuffle=True, #shuffle again to mix up game positions
            num_workers=0, # 0 for debugging, 2-4 for speedup later
            # drop_last=True # optional: drops last batch if less than 32
        )
        last_epoch = None
        for epoch in range(cfg.num_epochs):
            policy_train_loss = run_train_epoch(model, optimizer, train_loader, device, label="policy")
            value_train_loss = run_train_epoch(model, optimizer, train_loader, device, label="value")
            logger.info(f"Gen {gen} | Epoch {epoch} | Policy Loss: {policy_train_loss:.4f} | Value Loss: {value_train_loss:.4f}")
            last_epoch = epoch
        ckpt_path = save_checkpoint(
            model,
            optimizer,
            gen,
            cfg.checkpoint_dir,
            epoch=last_epoch,
            buffer_len=len(buffer),
            num_actions=4672,
        )
        logger.info(f"Saved checkpoint: {ckpt_path}")
        check_memory(logger, f"Gen {gen} End")
        logger.info("\n")

if __name__ == "__main__":
    main()
