import logging
import os

def setup_logger(log_dir="logs"):
    
    os.makedirs(log_dir,exist_ok=True)
    log_file = os.path.join(log_dir,"training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger()
