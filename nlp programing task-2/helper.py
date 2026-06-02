import logging
import torch
import random
import logging
import numpy as np
from datasets import Dataset, load_dataset

GROUP = "16" # TODO: Your group number here

def load_and_split_dataset(dataset_name: str = "tatsu-lab/alpaca",
                       small: bool = False, test_size=0.2) -> Dataset:
    """
    Load a local dataset from the specified path.
    Split the dataset into training and testing sets.
    Args:
        path: Path to the dataset directory
        small: If True, load a smaller subset of the dataset for testing purposes.
    """
    logger = logging.getLogger(__name__)  

    logger.info(f"Loading dataset: {dataset_name}")
    logger.info(f"Small dataset: {small}, Test size: {test_size}")
    
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Original dataset size: {len(dataset)}")
    
    dataset = dataset.select(range(10)) if small else dataset
    if small:
        logger.info(f"Selected subset size: {len(dataset)}")
    
    # TODO: split the dataset into train and test sets
    # Your code here
    split_data = dataset.train_test_split(test_size=test_size)
    logger.info(f"Split complete. Train: {len(split_data['train'])}, Test: {len(split_data['test'])}")
    dataset = split_data 
    
    # TODO: remove the "### Response:" part (i.e. true/gold output) from the text in the test set
    # Your code here
    for i in range(len(dataset['test'])):
        row = dataset['test'][i]
        instruction = row['instruction'].strip()
        input_text = row['input'].strip()
    
        if input_text:
           prompt = f"Instruction: {instruction}\nInput: {input_text}"
        else:
           prompt = f"Instruction: {instruction}"
    
        dataset['test'][i]['text'] = prompt.strip()
    logger.info("Removed response parts from test set")
    
    return dataset

def setup_logging(GROUP):
    """Set up logging configuration to log to both file and console"""
    log_filename = f"log_GROUP_{GROUP}.log"
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_filename, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger(__name__)

    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger


# For reproducibility, this should be at the very beginning of your script, do NOT change this
def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def _print(a,b):
    """Helper function"""
    logger = logging.getLogger(__name__)

    delim = "--" * 5
    message = f"{delim}\n{a}\n{b}\n{delim}"
    logger.info(message)
    print(message)