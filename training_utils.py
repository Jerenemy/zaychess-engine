import os
import torch
import logging
import torch.nn.functional as F
from torch.utils.data import DataLoader

from alpha_zero import (
    Config, 
    AlphaZeroNet, 
    setup_logger, 
)

cfg = Config()
logger = setup_logger('train', level=logging.DEBUG)

def save_checkpoint(model, optimizer, gen, epoch=None, buffer_len=None):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    name = f"az_gen_{gen}"
    if epoch is not None:
        name += f"_epoch_{epoch}"
    path = os.path.join(cfg.checkpoint_dir, f"{name}.pt")
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
    return path

def run_train_epoch(model: AlphaZeroNet, optimizer: torch.optim.Optimizer, dataloader: DataLoader, device: torch.device, target="policy"):
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
