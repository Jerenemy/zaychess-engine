from dataclasses import dataclass

@dataclass
class Config:
    # # Debugging Hyperparameters
    # num_gens: int = 3
    # num_epochs: int = 1
    # batch_size: int = 32
    # learning_rate: float = 0.01
    
    # # MCTS Settings
    # mcts_steps: int = 5
    # cpuct: float = 1.0
    
    # # Buffer Settings
    # max_buffer_size: int = 512
    # buffer_batch_size: int = 128
    
    # # Checkpoint Settings
    # checkpoint_dir: str = "checkpoints"
    # num_games: int = 50

    # Training Hyperparameters
    num_gens: int = 50
    num_epochs: int = 2
    batch_size: int = 32
    learning_rate: float = 0.01
    
    # MCTS Settings
    mcts_steps: int = 35
    cpuct: float = 1.0
    
    # Buffer Settings
    max_buffer_size: int = 3500
    buffer_batch_size: int = 2048
    
    # Checkpoint Settings
    checkpoint_dir: str = "checkpoints"
    num_games: int = 75