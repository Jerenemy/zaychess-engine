# ZayChess Engine (AlphaZero-style)

AlphaZero-style training and MCTS inference for chess and tic-tac-toe. The current layout is a small research codebase with a Python package (`alpha_zero`) plus several top-level scripts for training, evaluation, and UCI engine integration.

## Project layout

```
.
├── alpha_zero/                 # Core package
│   ├── __init__.py              # Public exports
│   ├── config.py                # Training/MCTS hyperparameters
│   ├── dataset.py               # Dataset + replay buffer + labeling
│   ├── logger_config.py         # Logger setup
│   ├── mcts.py                  # MCTS + Node implementation
│   ├── model.py                 # AlphaZero-style ResNet
│   ├── self_play.py             # Self-play loop
│   ├── utils.py                 # Action converters + tensorizers + helpers
│   ├── chess_wrapper.py         # Chess board wrapper
│   └── tictactoe.py             # Tic-tac-toe board implementation
├── tests/                       # Pytest/unittest coverage
├── notebooks/                   # Research notebook/script
├── checkpoints/                 # Saved model checkpoints (generated)
├── logs/                        # Logs (generated)
├── runs/                         # Experiment runs (generated)
├── evaluate.py                  # Evaluate model vs random baseline
├── play_ttt.py                  # Interactive tic-tac-toe
├── train.py                     # Chess training loop
├── train_ttt.py                 # Tic-tac-toe training loop
├── training_utils.py            # Shared training helpers
├── uci_engine.py                # UCI engine wrapper for chess
├── run_cutechess.sh             # UCI integration helper script
├── pyproject.toml               # Project metadata + dependencies
└── poetry.lock                  # Locked dependency versions
```

## Dependencies

- Python: `>=3.13,<3.15` (from `pyproject.toml`)
- Core packages: `torch`, `chess`, `numpy`, `psutil`, `pytest`

Install dependencies with one of the following:

```
poetry install
```

or

```
pip install -e .
```

## Key modules

- `alpha_zero/model.py`: ResNet-style policy/value network.
- `alpha_zero/mcts.py`: MCTS using model priors and value estimates.
- `alpha_zero/utils.py`: Action-space encoders and board tensorizers for chess/TTT.
- `alpha_zero/self_play.py`: Self-play data generation.
- `alpha_zero/dataset.py`: `AlphaZeroDataset`, `Buffer`, `label_data`.
- `alpha_zero/config.py`: Hyperparameters for training and self-play.

## Scripts

### Train (chess)
```
python train.py
```

### Train (tic-tac-toe)
```
python train_ttt.py
```

### Evaluate
```
python evaluate.py
```

Note: `evaluate.py` hard-codes the checkpoint path and game mode. Edit `game_mode` and checkpoint path as needed.

### Play tic-tac-toe
```
python play_ttt.py
```

### UCI engine
```
python uci_engine.py --checkpoint <path_to_checkpoint> [--mcts-steps N] [--cuda]
```

## Configuration

Defaults live in `alpha_zero/config.py`. You can override by editing that file or passing different values in code (e.g., `Config(mcts_steps=10, ...)`).

## Tests

```
pytest -q
```

## Notes

- Adapters (`ChessAdapter`, `TicTacToeAdapter`) provide converters/tensorizers; avoid global game-mode state.
- Checkpoints are written under `checkpoints/` and logs under `logs/` by default.
