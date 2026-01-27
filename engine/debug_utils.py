import sys
import psutil
import os


# 2. The Memory Checker
def check_memory(logger, step_name):
    """
    Logs current memory usage. 
    step_name: Where are we? (e.g. 'Start of MCTS', 'After Training')
    """
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3) # Convert bytes to GB
    
    logger.info(f"MEM CHECK [{step_name}]: {mem_gb:.2f} GB")
    
    # Optional: Safety tripwire
    if mem_gb > 16: # Adjust to your RAM limit
        logger.error("!!! MEMORY CRITICAL !!! Shutting down safely.")
        sys.exit(1)