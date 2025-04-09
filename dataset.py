"""
FineWeb-Edu dataset preprocessor for SRS pretraining.
Downloads, tokenizes, and shards the dataset to disk.

Dataset:
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

Usage:
    $ python fineweb.py

The script saves token shards to a local directory named "edu_fineweb10B".
"""

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset  # pip install datasets
from tqdm import tqdm  # pip install tqdm

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR = "edu_fineweb10B"
DATASET_VARIANT = "sample-10BT"  # dataset variant name
TOKENS_PER_SHARD = int(1e8)  # 100M tokens per shard; adjust as needed

# Create output directory if it does not exist.
output_dir = os.path.join(os.path.dirname(__file__), OUTPUT_DIR)
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Download dataset
# -----------------------------------------------------------------------------
# Load the FineWeb-Edu dataset (using the 'train' split).
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=DATASET_VARIANT, split="train")

# -----------------------------------------------------------------------------
# Tokenizer Initialization
# -----------------------------------------------------------------------------
# Initialize the GPT-2 tokenizer using tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
END_OF_TEXT_TOKEN = tokenizer._special_tokens['<|endoftext|>']


def tokenize_document(document):
    """
    Tokenizes a single document from the dataset.
    
    Each document is tokenized by first inserting the END_OF_TEXT_TOKEN as a delimiter,
    then encoding the document's text using GPT-2's tokenizer. The result is returned
    as a numpy array of uint16 tokens.
    
    Args:
        document (dict): A dictionary with a "text" field.
    
    Returns:
        np.ndarray: An array of uint16 token IDs.
    """
    # Start with a delimiter to mark the beginning/end of the document.
    tokens = [END_OF_TEXT_TOKEN]
    tokens.extend(tokenizer.encode_ordinary(document["text"]))
    token_array = np.array(tokens)
    
    # Ensure all token values are within the range for uint16.
    assert (0 <= token_array).all() and (token_array < 2**16).all(), "token dictionary too large for uint16"
    return token_array.astype(np.uint16)


def save_shard(filename, token_array):
    """
    Saves a numpy array of tokens to disk using the .npy format.
    
    Args:
        filename (str): Full path where the shard will be saved.
        token_array (np.ndarray): Array of tokens to save.
    """
    np.save(filename, token_array)


# -----------------------------------------------------------------------------
# Tokenization and Sharding Process
# -----------------------------------------------------------------------------
# Use half the available CPUs for tokenization.
num_processes = max(1, os.cpu_count() // 2)

# Allocate a buffer for the current shard.
current_shard = 0
buffer = np.empty((TOKENS_PER_SHARD,), dtype=np.uint16)
buffer_token_count = 0
progress_bar = None

# Create a pool for parallel tokenization.
with mp.Pool(num_processes) as pool:
    # Process documents in parallel with a reasonable chunksize.
    for token_array in pool.imap(tokenize_document, dataset, chunksize=16):
        # If the new token array fits entirely in the current shard buffer:
        if buffer_token_count + len(token_array) < TOKENS_PER_SHARD:
            buffer[buffer_token_count:buffer_token_count + len(token_array)] = token_array
            buffer_token_count += len(token_array)
            
            # Initialize progress bar for the current shard if not created.
            if progress_bar is None:
                progress_bar = tqdm(total=TOKENS_PER_SHARD, unit="tokens", desc=f"Shard {current_shard}")
            progress_bar.update(len(token_array))
        else:
            # Calculate how many tokens can fit in the current shard.
            tokens_to_fill = TOKENS_PER_SHARD - buffer_token_count
            progress_bar.update(tokens_to_fill)
            buffer[buffer_token_count:] = token_array[:tokens_to_fill]
            
            # Decide shard split type: use "val" for the first shard, then "train".
            shard_split = "val" if current_shard == 0 else "train"
            shard_filename = os.path.join(output_dir, f"edufineweb_{shard_split}_{current_shard:06d}.npy")
            
            # Save the filled shard.
            save_shard(shard_filename, buffer)
            current_shard += 1
            
            # Reset progress bar.
            progress_bar = None
            
            # The remaining tokens become the start of the next shard.
            remaining_tokens = token_array[tokens_to_fill:]
            remaining_length = len(remaining_tokens)
            buffer[:remaining_length] = remaining_tokens
            buffer_token_count = remaining_length

    # After processing all documents, save any tokens remaining in the buffer.
    if buffer_token_count != 0:
        shard_split = "val" if current_shard == 0 else "train"
        shard_filename = os.path.join(output_dir, f"edufineweb_{shard_split}_{current_shard:06d}.npy")
        save_shard(shard_filename, buffer[:buffer_token_count])
