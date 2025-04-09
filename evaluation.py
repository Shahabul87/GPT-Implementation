"""
Downloads and evaluates HellaSwag examples using GPT-2.

This script downloads the HellaSwag dataset (in JSONL format) from GitHub,
tokenizes each example using the GPT-2 tokenizer (via tiktoken), and then
evaluates model performance in a completion-style task.

Example HellaSwag JSON item:
{
  "ind": 24,
  "activity_label": "Roof shingle removal",
  "ctx_a": "A man is sitting on a roof.",
  "ctx_b": "he",
  "ctx": "A man is sitting on a roof. he",
  "split": "val",
  "split_type": "indomain",
  "label": 3,
  "endings": [
      "is using wrap to wrap a pair of skis.",
      "is ripping level tiles off.",
      "is holding a rubik's cube.",
      "starts pulling up roofing on a roof."
  ],
  "source_id": "activitynet~v_-JhWjGDPHMY"
}

Usage:
    $ python hellaswag.py -m gpt2 -d cuda
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")
HELLASWAG_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# Create cache directory if it does not exist.
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Initialize GPT-2 tokenizer using tiktoken.
tokenizer = tiktoken.get_encoding("gpt2")


# -----------------------------------------------------------------------------
# Helper Functions: Downloading
# -----------------------------------------------------------------------------
def download_file(url: str, filename: str, chunk_size: int = 1024) -> None:
    """
    Downloads a file from the given URL and saves it to the specified filename.
    
    Args:
        url (str): The URL to download the file from.
        filename (str): The local path where the file should be saved.
        chunk_size (int): Number of bytes to read at a time.
    """
    response = requests.get(url, stream=True)
    total_bytes = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as file, tqdm(
        desc=f"Downloading {os.path.basename(filename)}",
        total=total_bytes,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            progress_bar.update(size)


def download_data(split: str) -> None:
    """
    Downloads the HellaSwag dataset for the given split (train, val, or test)
    and saves it to the cache directory if not already present.
    
    Args:
        split (str): One of 'train', 'val', or 'test'.
    """
    data_url = HELLASWAG_URLS[split]
    local_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(local_filename):
        print(f"Downloading {split} split from {data_url}...")
        download_file(data_url, local_filename)


# -----------------------------------------------------------------------------
# Helper Functions: Data Preparation for Evaluation
# -----------------------------------------------------------------------------
def render_example(example: dict):
    """
    Renders a HellaSwag example into tokens, attention mask, and label.
    
    It processes the context and each candidate ending (after prepending a space
    for the GPT-2 tokenizer) to produce a list of token sequences and corresponding
    masks. The mask distinguishes between the context (0's) and the candidate ending (1's).
    
    Args:
        example (dict): A HellaSwag example dictionary.
    
    Returns:
        tuple: A tuple containing:
            - meta (dict): A dictionary with metadata ("label", "ctx_tokens", "ending_tokens").
            - tokens (torch.Tensor): A tensor of shape (4, max_seq_len) containing token IDs.
            - mask (torch.Tensor): A tensor of shape (4, max_seq_len) indicating candidate region.
            - label (int): The index of the correct ending.
    """
    context = example["ctx"]
    correct_label = example["label"]
    candidate_endings = example["endings"]

    meta = {
        "label": correct_label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # Tokenize the context.
    context_tokens = tokenizer.encode(context)
    meta["ctx_tokens"] = context_tokens

    token_rows = []
    mask_rows = []
    for ending in candidate_endings:
        # Prepend a space for correct tokenization.
        ending_tokens = tokenizer.encode(" " + ending)
        token_rows.append(context_tokens + ending_tokens)
        mask_rows.append([0] * len(context_tokens) + [1] * len(ending_tokens))
        meta["ending_tokens"].append(ending_tokens)

    # Find the maximum row length for padding.
    max_length = max(len(row) for row in token_rows)
    tokens_tensor = torch.zeros((4, max_length), dtype=torch.long)
    mask_tensor = torch.zeros((4, max_length), dtype=torch.long)

    for idx, (row_tokens, row_mask) in enumerate(zip(token_rows, mask_rows)):
        tokens_tensor[idx, :len(row_tokens)] = torch.tensor(row_tokens)
        mask_tensor[idx, :len(row_mask)] = torch.tensor(row_mask)

    return meta, tokens_tensor, mask_tensor, correct_label


def iterate_examples(split: str):
    """
    Yields HellaSwag examples from a given split as Python dictionaries.
    
    Args:
        split (str): One of 'train', 'val', or 'test'.
    
    Yields:
        dict: A HellaSwag example read from the corresponding JSONL file.
    """
    download_data(split)
    file_path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(file_path, "r") as f:
        for line in f:
            yield json.loads(line)


# -----------------------------------------------------------------------------
# Evaluation Function
# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model_type: str, device: str) -> None:
    """
    Loads a GPT-2 model, evaluates it on the HellaSwag validation set,
    and prints the accuracy statistics.
    
    The evaluation is done by rendering each example, computing the average
    autoregressive loss over the candidate completion region, and selecting
    the candidate with the lowest loss.
    
    Args:
        model_type (str): The name of the model to load (e.g., "gpt2").
        device (str): The device to run the evaluation on (e.g., "cuda" or "cpu").
    """
    # Set high matmul precision (e.g., using TF32 on supported CUDA hardware)
    torch.set_float32_matmul_precision('high')
    
    # Load and move the model to device.
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # Uncomment the following line to use torch.compile if desired.
    # model = torch.compile(model)
    
    total_correct = 0
    total_correct_norm = 0
    total_examples = 0
    
    for example in iterate_examples("val"):
        meta, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get model output logits.
        outputs = model(tokens)
        logits = outputs.logits

        # Compute autoregressive loss for each candidate completion.
        # Shift logits and tokens to align predictions with targets.
        shifted_logits = logits[..., :-1, :].contiguous()
        shifted_tokens = tokens[..., 1:].contiguous()
        flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
        flat_targets = shifted_tokens.view(-1)
        token_loss = F.cross_entropy(flat_logits, flat_targets, reduction='none')
        token_loss = token_loss.view(tokens.size(0), -1)

        # Adjust mask for shifted tokens.
        shifted_mask = mask[..., 1:].contiguous()
        masked_losses = token_loss * shifted_mask

        # Calculate average loss per candidate (completion region only).
        loss_sum = masked_losses.sum(dim=1)
        avg_loss = loss_sum / shifted_mask.sum(dim=1)

        # Predicted indices: lowest raw and normalized loss.
        pred_raw = loss_sum.argmin().item()
        pred_norm = avg_loss.argmin().item()

        total_examples += 1
        total_correct += int(pred_raw == label)
        total_correct_norm += int(pred_norm == label)

        acc = total_correct / total_examples
        acc_norm = total_correct_norm / total_examples

        print(f"Processed {total_examples} examples | acc: {acc:.4f} | acc_norm: {acc_norm:.4f}")

        # Debug: print first few examples with details.
        if total_examples < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print("Endings:")
            for i, ending in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) -> {ending}")
            print(f"Predicted (normalized): {pred_norm}, Actual: {label}\n")
    
    print(f"Final normalized accuracy: {acc_norm:.4f}")


# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HellaSwag using GPT-2")
    parser.add_argument("-m", "--model_type", type=str, default="gpt2",
                        help="Name of the GPT model type to use (default: gpt2)")
    parser.add_argument("-d", "--device", type=str, default="cuda",
                        help="Device to run the model on (e.g., cuda or cpu)")
    args = parser.parse_args()

    evaluate(args.model_type, args.device)
