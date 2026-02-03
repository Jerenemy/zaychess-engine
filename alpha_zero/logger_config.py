from __future__ import annotations

import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Sets up a logger that outputs to console and a timestamped file.
    Args:
        name: Name of the logger (e.g., "MCTS", "Train")
        log_dir: Folder to store logs (default: "logs")
        level: Logging level (DEBUG, INFO, etc.)
    """
    # 1. Create the log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 2. Create a unique filename using the current time
    # Format: logs/run_2023-10-27_14-30-01.log
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f"run_{timestamp}.log")

    # 3. Setup the logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate logs if function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # 4. Create Handlers
    c_handler = logging.StreamHandler(sys.stdout) # Console
    f_handler = logging.FileHandler(log_file)     # File

    # 5. Create Formatters
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    # We don't need date in file log anymore since filename has it, but it's good for long runs
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # 6. Add Handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    # Optional: Print where the log is saved so you know
    print(f"Logging to: {log_file}")

    return logger
