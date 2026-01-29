from dataclasses import dataclass

@dataclass
class Config:
    # # Debugging Hyperparameters
    num_gens: int = 20
    num_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # MCTS Settings
    mcts_steps: int = 40
    cpuct: float = 1.0
    max_num_moves_per_game: int = 20
    temperature_move_threshold: int = 2
    
    # Buffer Settings
    max_buffer_size: int = 1024
    buffer_batch_size: int = 256
    
    # Checkpoint Settings
    checkpoint_dir: str = "checkpoints"
    num_games_per_gen: int = 20

    # # Training Hyperparameters
    # num_gens: int = 50
    # num_epochs: int = 2
    # batch_size: int = 32
    # learning_rate: float = 0.01
    
    # # MCTS Settings
    # mcts_steps: int = 35
    # cpuct: float = 1.0
    # max_num_moves_per_game: int = 200
    # temperature_move_threshold: int = 30
    
    # # Buffer Settings
    # max_buffer_size: int = 3500
    # buffer_batch_size: int = 2048
    
    # # Checkpoint Settings
    # checkpoint_dir: str = "checkpoints"
    # num_games: int = 75
