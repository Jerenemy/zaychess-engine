import torch
import logging
from torch.utils.data import DataLoader

from alpha_zero import Config, AlphaZeroNet, play_one_game, AlphaZeroDataset, Buffer, TicTacToeAdapter
from alpha_zero.logger_config import setup_logger
from training_utils import run_train_epoch, save_checkpoint

def main():
    cfg = Config()
    logger = setup_logger(name='train_ttt', level=logging.DEBUG)

    model = AlphaZeroNet(input_shape=(2, 3, 3), num_actions=9, num_resblocks=5)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    buffer = Buffer(maxlen=cfg.max_buffer_size)
    adapter = TicTacToeAdapter()

    for gen in range(cfg.num_gens):
        gen_data = []
        # play games
        for game in range(cfg.num_games_per_gen):
            game_data, _, _ = play_one_game(model, cfg, adapter=adapter, logger=logger)
            gen_data.extend(game_data)
        # add to buffer
        buffer.add(gen_data)
        raw_batch = buffer.sample_batch(cfg.buffer_batch_size) # list of raw tuples: (s1, p1, v1), (s2, p2, v2), ...
        # construct dataset and dataloader
        dataset = AlphaZeroDataset(raw_batch)
        train_loader = DataLoader(
            dataset, 
            batch_size=cfg.batch_size, #mini batch size for gpu
            shuffle=True, #shuffle again to mix up game positions
            num_workers=0, # 0 for debugging, 2-4 for speedup later
            # drop_last=True # optional: drops last batch if less than 32
        )
        # train
        last_epoch = None
        for epoch in range(cfg.num_epochs):
            p_loss = run_train_epoch(model, optimizer, train_loader, device, target='policy')
            v_loss = run_train_epoch(model, optimizer, train_loader, device, target='value')
            logger.info(f"epoch {epoch}, policy loss: {p_loss:.4}, value loss {v_loss:.4}")
            last_epoch = epoch
        # save checkpoint
        ckpt_path = save_checkpoint(
            model,
            optimizer,
            gen,
            cfg.checkpoint_dir,
            epoch=last_epoch,
            buffer_len=len(buffer),
            num_actions=9,
        )
        logger.info(f"saved checkpoint at {ckpt_path}")
        logger.info(f"Gen {gen} end.\n")
        logger.debug(f"buffer len: {len(buffer)}")

if __name__ == "__main__":
    main()
    
    
