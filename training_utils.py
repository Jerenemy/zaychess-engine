from __future__ import annotations

from typing import Optional, Callable, Any
import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alpha_zero import AlphaZeroNet, Buffer, AlphaZeroDataset, play_one_game
from alpha_zero.config import Config
from alpha_zero.game_adapter import GameAdapter

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

def _next_run_id(base_dir: str, game_tag: str) -> int:
    prefix = f"{game_tag}_run_"
    max_id = 0
    if not os.path.exists(base_dir):
        return 1
    for name in os.listdir(base_dir):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.isdigit():
            max_id = max(max_id, int(suffix))
    return max_id + 1

def run_train_epoch(
    model: AlphaZeroNet,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    target: str = "policy",
) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        X = batch["state"].to(device)
        policy_logits_pred, value_pred = model(X)
        if target == "policy":
            y_true = batch["policy"].to(device)
            log_probs = F.log_softmax(policy_logits_pred, dim=1)
            loss = -torch.sum(y_true * log_probs, dim=1).mean()
        elif target == "value":
            y_true = batch["value"].to(device).unsqueeze(1)
            loss = F.mse_loss(value_pred, y_true)
        else: 
            raise ValueError("Invalid target, either policy or value")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def run_self_play_training(
    model: AlphaZeroNet,
    cfg: Config,
    adapter: GameAdapter,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    check_memory_fn: Optional[Callable[[logging.Logger, str], None]] = None,
    game_tag: str = "chess",
) -> None:
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    run_id = _next_run_id(cfg.checkpoint_dir, game_tag)
    run_dir = os.path.join(cfg.checkpoint_dir, f"{game_tag}_run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Checkpoint run dir: {run_dir}")

    buffer = Buffer(maxlen=cfg.max_buffer_size)
    logger.info(f"using device: {device}")
    logger.info(
        "num_gens: %s, num_epochs: %s, mcts_steps: %s, buffer_size: %s, buffer_batch_size: %s\n",
        cfg.num_gens,
        cfg.num_epochs,
        cfg.mcts_steps,
        buffer.maxlen,
        cfg.buffer_batch_size,
    )

    for gen in range(cfg.num_gens):
        logger.info(f"Starting generation {gen}")
        if check_memory_fn is not None:
            check_memory_fn(logger, "Start of Generation")
        logger.info(f"Buffer length: {len(buffer)}")
        new_gen_data: list[tuple[Any, Any, float]] = []

        # play games
        for game in range(cfg.num_games_per_gen):
            new_game_data, result_str, num_moves = play_one_game(
                model,
                cfg,
                adapter=adapter,
                logger=logger,
            )
            logger.info(f"Game {game} | Game result: {result_str} ({num_moves} moves)")
            new_gen_data.extend(new_game_data)
            if check_memory_fn is not None:
                check_memory_fn(logger, f"After game {game}")

        buffer.add(new_gen_data)

        # train
        raw_batch = buffer.sample_batch(cfg.buffer_batch_size)
        dataset = AlphaZeroDataset(raw_batch)
        train_loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
        )
        last_epoch = None
        for epoch in range(cfg.num_epochs):
            policy_train_loss = run_train_epoch(model, optimizer, train_loader, device, target="policy")
            value_train_loss = run_train_epoch(model, optimizer, train_loader, device, target="value")
            logger.info(
                "Gen %s | Epoch %s | Policy Loss: %.4f | Value Loss: %.4f",
                gen,
                epoch,
                policy_train_loss,
                value_train_loss,
            )
            last_epoch = epoch

        ckpt_path = save_checkpoint(
            model,
            optimizer,
            gen,
            run_dir,
            epoch=last_epoch,
            buffer_len=len(buffer),
            num_actions=adapter.action_space_size,
        )
        logger.info(f"Saved checkpoint: {ckpt_path}")
        if check_memory_fn is not None:
            check_memory_fn(logger, f"Gen {gen} End")
        logger.info("\n")
