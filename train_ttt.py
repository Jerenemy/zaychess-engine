from __future__ import annotations

import logging
import torch

from alpha_zero import Config, AlphaZeroNet, TicTacToeAdapter
from alpha_zero.logger_config import setup_logger
from training_utils import run_self_play_training

def main() -> None:
    cfg = Config()
    logger = setup_logger(name='train_ttt', level=logging.DEBUG)

    adapter = TicTacToeAdapter()
    model = AlphaZeroNet(
        input_shape=adapter.input_shape,
        num_actions=adapter.action_space_size,
        num_resblocks=5,
    )
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    run_self_play_training(
        model=model,
        cfg=cfg,
        adapter=adapter,
        optimizer=optimizer,
        device=device,
        logger=logger,
        game_tag="ttt",
    )

if __name__ == "__main__":
    main()
    
    
