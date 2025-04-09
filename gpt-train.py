import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from evaluation import render_example, iterate_examples


class CausalSelfAttention(nn.Module):
    """
    Implements a multi-head causal self-attention mechanism.
    
    This module projects the input embeddings into query, key, and value representations,
    applies causal (masked) scaled dot-product attention, and then projects the results back.
    
    Attributes:
        qkv_proj (nn.Linear): Combined linear projection for queries, keys, and values.
        out_proj (nn.Linear): Linear projection for the concatenated attention output.
        num_heads (int): Number of attention heads.
        embed_dim (int): Total embedding dimensionality.
    """
    
    def __init__(self, config):
        """
        Initializes the CausalSelfAttention module.
        
        Args:
            config: Configuration object with attributes `embed_dim` and `num_heads`.
                    It is assumed that embed_dim is divisible by num_heads.
        """
        super().__init__()
        
        # Use the configuration parameters defined in GPTConfig
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        
        # Ensure that the embedding dimension is divisible by the number of heads.
        assert self.embed_dim % self.num_heads == 0, (
            "The embedding dimension must be divisible by the number of heads."
        )
        
        # Linear projection for queries, keys, and values in a single matrix multiplication.
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim)
        
        # Linear projection to combine the output of all attention heads.
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, inputs):
        """
        Forward pass for the causal self-attention module.
        
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_length, embed_dim).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, embed_dim).
        """
        batch_size, seq_length, _ = inputs.size()
        
        # Project the inputs to obtain queries, keys, and values.
        qkv = self.qkv_proj(inputs)  # Shape: (batch_size, seq_length, 3 * embed_dim)
        
        # Split the projection into separate query, key, and value tensors.
        # Each will have shape: (batch_size, seq_length, embed_dim)
        query, key, value = qkv.split(self.embed_dim, dim=2)
        
        # Reshape and transpose so that the head dimension comes forward.
        head_dim = self.embed_dim // self.num_heads
        # New shape: (batch_size, num_heads, seq_length, head_dim)
        query = query.view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention with causal masking.
        # Output shape: (batch_size, num_heads, seq_length, head_dim)
        attention_output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        
        # Transpose and reshape to concatenate the multiple heads.
        # New shape: (batch_size, seq_length, embed_dim)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.embed_dim)
        
        # Final linear projection.
        output = self.out_proj(attention_output)
        
        return output




class MLP(nn.Module):
    """
    Multilayer Perceptron (MLP) used as a feed-forward network in transformer architectures.

    This module applies a two-layer fully connected feed-forward network with
    a GELU activation function (using the tanh approximation) between the layers.
    It first expands the embedding dimensionality by a factor of 4 and then projects
    it back to the original embedding size.

    Attributes:
        fc_layer (nn.Linear): Linear layer that expands the embedding.
        activation (nn.GELU): GELU activation function with tanh approximation.
        projection_layer (nn.Linear): Linear layer that projects back to the original embedding dimension.
    """

    def __init__(self, config):
        """
        Initialize the MLP module.

        Args:
            config: Configuration object containing model hyperparameters.
                    It is expected to have the attribute `embed_dim`.
        """
        super().__init__()
        
        # Retrieve the embedding dimension from the configuration.
        embed_dim = config.embed_dim
        hidden_dim = 4 * embed_dim  # Common practice is to use a 4x expansion in transformer architectures.
        
        # Define the first fully connected layer to expand the embedding dimension.
        self.fc_layer = nn.Linear(embed_dim, hidden_dim)
        # Use a GELU activation function with the tanh approximation for non-linearity.
        self.activation = nn.GELU(approximate="tanh")
        # Define the projection layer to project back to the original embedding dimension.
        self.projection_layer = nn.Linear(hidden_dim, embed_dim)
        # Custom attribute for scaling initialization (e.g., used in NanoGPT).
        self.projection_layer.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        """
        Forward pass of the MLP module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embed_dim).
        """
        # Expand the embedding dimension.
        x = self.fc_layer(x)
        # Apply GELU activation.
        x = self.activation(x)
        # Project back to the original embedding dimension.
        x = self.projection_layer(x)
        return x






class TransformerBlock(nn.Module):
    """
    TransformerBlock implements a single block of a GPT-style model.
    
    Each block consists of:
      1. A Layer Normalization followed by a causal self-attention module with a residual connection.
      2. A second Layer Normalization followed by a feed-forward MLP (multilayer perceptron) with a residual connection.
    """
    
    def __init__(self, config):
        """
        Initializes the TransformerBlock.
        
        Args:
            config: A configuration object containing model hyperparameters.
                    Expected attributes:
                        - embed_dim (int): Embedding dimension.
                        - num_heads (int): Number of attention heads.
        """
        super().__init__()
        # Pre-attention normalization.
        self.norm1 = nn.LayerNorm(config.embed_dim)
        # Causal self-attention module.
        self.attention = CausalSelfAttention(config)
        # Pre-MLP normalization.
        self.norm2 = nn.LayerNorm(config.embed_dim)
        # Feed-forward MLP.
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Executes the forward pass of the transformer block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embed_dim).
            
        Returns:
            torch.Tensor: Output tensor of the same shape as the input.
        """
        # Apply layer normalization and causal self-attention with residual connection.
        x = x + self.attention(self.norm1(x))
        # Apply layer normalization and MLP with residual connection.
        x = x + self.mlp(self.norm2(x))
        return x

@dataclass
class GPTConfig:
    """
    Configuration parameters for a GPT-style Transformer model.
    
    Attributes:
        block_size (int): Maximum sequence length (context window).
        vocab_size (int): Number of unique tokens in the vocabulary.
        num_layers (int): Number of transformer blocks (layers) in the model.
        num_heads (int): Number of attention heads per transformer block.
        embed_dim (int): Dimensionality of the token embeddings.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    embed_dim: int = 768



class GPT(nn.Module):
    """
    GPT Transformer model that incorporates token embeddings, positional embeddings, 
    a stack of transformer blocks, and a final projection layer for language modeling.
    
    Attributes:
        config: A configuration object containing model hyperparameters.
        token_embedding (nn.Embedding): Embedding layer for input tokens.
        position_embedding (nn.Embedding): Embedding layer for token positions.
        transformer_layers (nn.ModuleList): List of transformer blocks.
        final_layer_norm (nn.LayerNorm): Final layer normalization applied before the classifier.
        lm_head (nn.Linear): Linear layer that projects hidden states to vocabulary logits.
    """
    
    def __init__(self, config):
        """
        Initialize the GPT model.
        
        Args:
            config: Configuration object with the following expected attributes:
                - vocab_size: Size of the vocabulary.
                - block_size: Maximum sequence length.
                - num_layers: Number of transformer blocks.
                - num_heads: Number of attention heads.
                - embed_dim: Embedding dimensionality.
        """
        super().__init__()
        self.config = config

        # Token and positional embeddings.
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.block_size, config.embed_dim)
        
        # Transformer blocks.
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer normalization.
        self.final_layer_norm = nn.LayerNorm(config.embed_dim)
        
        # Language modeling head. Weight sharing is applied with token_embedding.
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight  # Weight tying
        
        # Initialize model parameters.
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        
        Args:
            module (nn.Module): A module instance whose parameters will be initialized.
        """
        if isinstance(module, nn.Linear):
            # Standard deviation for initialization.
            std = 0.02
            # Adjust standard deviation based on NanoGPT scaling if applicable.
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.num_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        """
        Execute a forward pass through the GPT model.
        
        Args:
            input_ids (torch.Tensor): Input token indices of shape (batch_size, sequence_length).
            targets (torch.Tensor, optional): Ground truth indices for computing loss.
        
        Returns:
            logits (torch.Tensor): Logits over the vocabulary with shape (batch_size, sequence_length, vocab_size).
            loss (torch.Tensor or None): Cross entropy loss computed over the entire batch if targets are provided, else None.
        """
        batch_size, seq_length = input_ids.size()
        if seq_length > self.config.block_size:
            raise ValueError(
                f"Sequence length {seq_length} exceeds model block size {self.config.block_size}."
            )
        
        # Create positional indices and compute positional embeddings.
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        pos_embeddings = self.position_embedding(positions)  # (seq_length, embed_dim)
        
        # Compute token embeddings and add positional embeddings.
        token_embeddings = self.token_embedding(input_ids)  # (batch_size, seq_length, embed_dim)
        x = token_embeddings + pos_embeddings  # Broadcasting over batch dimension
        
        # Pass through the stacked transformer layers.
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply final layer normalization.
        x = self.final_layer_norm(x)
        
        # Generate logits for each token.
        logits = self.lm_head(x)  # (batch_size, seq_length, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten the logits and targets for cross entropy loss computation.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss



def configure_optimizers(self, weight_decay, learning_rate, device_type):
    """
    Configures the AdamW optimizer for a single GPU or CPU setup by grouping parameters
    based on whether weight decay should be applied.
    
    This method organizes the model parameters into two groups:
      1. Parameters with 2 or more dimensions (typically weight matrices and embeddings) 
         which will have weight decay applied.
      2. Parameters with fewer than 2 dimensions (biases and layer normalization parameters)
         which will not have weight decay applied.
    
    Args:
        weight_decay (float): Weight decay factor for the eligible parameters.
        learning_rate (float): Learning rate for the optimizer.
        device_type (str): Device type, e.g., "cuda" for GPU or "cpu" for CPU.
    
    Returns:
        torch.optim.Optimizer: An instance of AdamW configured with the parameter groups.
    """
    # Gather all trainable parameters.
    parameters = {name: param for name, param in self.named_parameters() if param.requires_grad}

    # Split parameters: those with 2 or more dimensions get weight decay.
    decay_parameters = [param for param in parameters.values() if param.dim() >= 2]
    no_decay_parameters = [param for param in parameters.values() if param.dim() < 2]

    # Create parameter groups with appropriate weight decay settings.
    optimizer_param_groups = [
        {'params': decay_parameters, 'weight_decay': weight_decay},
        {'params': no_decay_parameters, 'weight_decay': 0.0}
    ]

    # Optionally log the parameter counts and totals.
    num_decay_params = sum(param.numel() for param in decay_parameters)
    num_no_decay_params = sum(param.numel() for param in no_decay_parameters)
    print(f"Number of weight-decayed parameter tensors: {len(decay_parameters)}, total parameters: {num_decay_params:,}")
    print(f"Number of non-weight-decayed parameter tensors: {len(no_decay_parameters)}, total parameters: {num_no_decay_params:,}")

    # Determine if the fused AdamW optimizer can be used (available if on CUDA and supported).
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and (device_type == "cuda")
    print(f"Using fused AdamW: {use_fused}")

    # Create and return the AdamW optimizer.
    optimizer = torch.optim.AdamW(
        optimizer_param_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused
    )
    return optimizer


import os
import torch
import numpy as np
import tiktoken
from typing import Tuple  # Import Tuple for type annotations

def load_tokens(filename: str) -> torch.Tensor:
    """
    Loads tokenized data from a .npy file and converts it to a torch.Tensor.
    
    Args:
        filename (str): Path to the .npy file containing tokenized data.
    
    Returns:
        torch.Tensor: Tensor of token indices with dtype torch.long.
    """
    np_tokens = np.load(filename)
    np_tokens = np_tokens.astype(np.int32)  # Ensure proper type conversion
    token_tensor = torch.tensor(np_tokens, dtype=torch.long)
    return token_tensor

class DataLoaderLite:
    """
    A lightweight data loader to iterate over tokenized data stored in shards.

    This loader manages a list of shard files for a given dataset split ('train' or 'val')
    and provides batches for language modeling by slicing the tokenized data.

    Attributes:
        batch_size (int): Number of samples (sequences) per batch.
        seq_length (int): Number of tokens per sequence (context window).
        split (str): Specifies the dataset split, must be either 'train' or 'val'.
        shards (list[str]): Full paths to token shard files.
        tokens (torch.Tensor): Tokenized data from the current shard.
        current_shard (int): Index of the currently loaded shard.
        current_position (int): Current reading position in the token stream.
    """
    def __init__(self, batch_size: int, seq_length: int, split: str) -> None:
        """
        Initializes the DataLoaderLite.

        Args:
            batch_size (int): Number of samples per batch.
            seq_length (int): Sequence length (context window) for each sample.
            split (str): Must be either 'train' or 'val'.
        """
        if split not in {'train', 'val'}:
            raise ValueError("`split` must be either 'train' or 'val'")

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split

        # Define the root directory containing shard files.
        data_root = "edu_fineweb10B"
        shard_files = os.listdir(data_root)
        # Filter shards according to the desired split.
        shard_files = [s for s in shard_files if split in s]
        shard_files = sorted(shard_files)
        # Full paths to each shard file.
        self.shards = [os.path.join(data_root, s) for s in shard_files]
        if len(self.shards) == 0:
            raise ValueError(f"No shards found for split '{split}'")
        # For single-device setup, always log this information.
        print(f"Found {len(self.shards)} shards for split '{split}'")

        self.reset()

    def reset(self) -> None:
        """
        Resets the data loader to start from the beginning of the first shard.
        """
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the next batch of tokenized data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (x, y) where:
                - x has shape (batch_size, seq_length) as input tokens.
                - y has shape (batch_size, seq_length) as target tokens (x shifted by one token).
        """
        B = self.batch_size
        T = self.seq_length
        # Calculate number of tokens needed for a full batch plus one extra token for targets.
        total_tokens_needed = B * T + 1
        buffer = self.tokens[self.current_position : self.current_position + total_tokens_needed]

        # If there are not enough tokens in the current shard, load the next shard.
        if buffer.size(0) < total_tokens_needed:
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
            buffer = self.tokens[self.current_position : self.current_position + total_tokens_needed]
            if buffer.size(0) < total_tokens_needed:
                raise ValueError("Not enough tokens in the shard to create a full batch.")

        # Create input and target sequences.
        x = buffer[:-1].view(B, T)  # Inputs.
        y = buffer[1:].view(B, T)   # Targets (shifted by one token).

        # Update the current reading position for the next batch.
        self.current_position += B * T
        return x, y


import torch
import torch.nn.functional as F

def get_most_likely_completion(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int:
    """
    Computes the average autoregressive loss over the completion region for each example,
    and returns the index of the example with the lowest average loss.

    The function shifts logits and tokens to align predictions with targets,
    computes the cross-entropy loss per token, averages the loss over positions marked
    in the provided mask (excluding the prompt region), and selects the example with
    the minimum average loss.

    Args:
        tokens (torch.Tensor): Input token indices with shape (batch_size, sequence_length).
        mask (torch.Tensor): Binary mask with shape (batch_size, sequence_length) where a '1'
                             indicates positions to be evaluated as the "completion" region.
        logits (torch.Tensor): Model logits with shape (batch_size, sequence_length, vocab_size).

    Returns:
        int: The index of the completion row with the lowest average loss.
    """
    # Shift logits and tokens to align predictions with target tokens.
    shifted_logits = logits[..., :-1, :].contiguous()  # Exclude the last logit.
    shifted_tokens = tokens[..., 1:].contiguous()        # Exclude the first token (prompt).

    # Flatten logits and tokens to calculate cross-entropy loss per token.
    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_tokens = shifted_tokens.view(-1)

    # Compute per-token cross entropy loss (no reduction).
    per_token_loss = F.cross_entropy(flat_logits, flat_tokens, reduction='none')
    per_token_loss = per_token_loss.view(tokens.size(0), -1)  # Shape: (batch_size, sequence_length - 1)

    # Adjust the mask for the shifted tokens (exclude the prompt token).
    shifted_mask = mask[..., 1:].contiguous()

    # Compute masked losses where only the "completion" tokens (mask == 1) contribute.
    masked_loss = per_token_loss * shifted_mask

    # Average loss per example: sum losses and divide by the count of active mask positions.
    loss_sum = masked_loss.sum(dim=1)
    avg_loss = loss_sum / shifted_mask.sum(dim=1)

    # Return the index of the example with the lowest average loss.
    return avg_loss.argmin().item()


import os
import math
import torch
import tiktoken

# =============================================================================
# 1. Device Configuration
# =============================================================================
# Determine the device to run on: prefer CUDA, then MPS (Apple Silicon) and finally CPU.
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# For consistency with other parts of the code, set device_type for weight decay checks
device_type = "cuda" if device.startswith("cuda") else "cpu"

# =============================================================================
# 2. Reproducibility: Set Random Seeds
# =============================================================================
seed = 1337
torch.manual_seed(seed)
if device == "cuda":
    torch.cuda.manual_seed(seed)
enc = tiktoken.get_encoding("gpt2")
# =============================================================================
# 3. Token Encoding Setup
# =============================================================================
# Retrieve the GPT-2 encoding (tiktoken provides a way to tokenize text in the same way as GPT-2)
encoder = tiktoken.get_encoding("gpt2")

# =============================================================================
# 4. Training Hyperparameters and Batch Setup
# =============================================================================
total_batch_tokens = 524288  # Total desired batch size measured in tokens (~0.5M tokens)
micro_batch_size = 64        # Micro batch size (number of samples per batch)
seq_length = 1024            # Sequence length (context window)

# Ensure the total batch size is divisible by (micro_batch_size * seq_length)
assert total_batch_tokens % (micro_batch_size * seq_length) == 0, \
    "total_batch_tokens must be divisible by micro_batch_size * seq_length"

# Calculate how many gradient accumulation steps are required to reach total_batch_tokens.
grad_accum_steps = total_batch_tokens // (micro_batch_size * seq_length)
print(f"Total desired batch size (in tokens): {total_batch_tokens}")
print(f"Calculated gradient accumulation steps: {grad_accum_steps}")

# =============================================================================
# 5. Data Loader Initialization
# =============================================================================
# For a single-GPU/CPU run, process_rank is 0 and num_processes is 1.
# DataLoaderLite is assumed to be defined elsewhere.
train_loader = DataLoaderLite(batch_size=micro_batch_size, seq_length=seq_length, split="train")
val_loader   = DataLoaderLite(batch_size=micro_batch_size, seq_length=seq_length, split="val")


# =============================================================================
# 6. Set Matmul Precision
# =============================================================================
# This setting helps to ensure high precision for 32-bit floating point matrix multiplications.
torch.set_float32_matmul_precision('high')

# =============================================================================
# 7. Model Instantiation and Setup
# =============================================================================
# Create a GPT model instance with the desired configuration.
model_config = GPTConfig(vocab_size=50304)
model = GPT(model_config)
model.to(device)

# Optionally compile the model for performance (disabled by default)
use_compile = False  # torch.compile may interfere with certain evaluations; disable if issues occur.
if use_compile:
    model = torch.compile(model)

# For single-device runs, the raw model is not wrapped in DistributedDataParallel.
raw_model = model

# =============================================================================
# 8. Learning Rate Scheduler Setup
# =============================================================================
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073  # For example, this many steps may approximate one epoch for a 10B-token dataset.

def get_lr(step):
    """
    Compute the learning rate for a given training step using linear warmup followed by cosine decay.
    
    Args:
        step (int): Current training step.
        
    Returns:
        float: The calculated learning rate.
    """
    # Linear warmup for the initial warmup_steps
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # After max_steps, the learning rate is fixed at min_lr.
    if step > max_steps:
        return min_lr
    # Otherwise, apply cosine decay from max_lr down to min_lr.
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1, "decay_ratio is out of bounds"
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + cosine_decay * (max_lr - min_lr)

# =============================================================================
# 9. Optimizer Configuration
# =============================================================================
# The model’s own method 'configure_optimizers' is used to set up AdamW with weight decay.
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, device_type=device_type)

# At this point, the setup is complete: the device, random seeds, data loaders, model, 
# learning rate scheduler, and optimizer are ready for the training loop.


import os
import time
import math
import torch

# =============================================================================
# 1. Setup Logging Directory
# =============================================================================
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "log.txt")

# Clear previous logs
with open(log_file_path, "w") as log_file:
    pass

# =============================================================================
# 2. Training Loop with Validation and Checkpointing
# =============================================================================
for step in range(max_steps):
    start_time = time.time()
    is_last_step = (step == max_steps - 1)

    # -------------------------------------------------------------------------
    # 2a. Periodic Validation Evaluation
    # -------------------------------------------------------------------------
    # Evaluate model on the validation set every 250 steps (or on the last step)
    if step % 250 == 0 or is_last_step:
        model.eval()  # set model to evaluation mode
        val_loader.reset()  # restart the validation loader

        # Accumulate validation loss over a fixed number of steps
        num_val_steps = 20
        validation_loss_total = 0.0

        with torch.no_grad():
            for _ in range(num_val_steps):
                x_val, y_val = val_loader.next_batch()
                x_val, y_val = x_val.to(device), y_val.to(device)

                # Use autocasting for performance and reduced memory consumption
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits_val, loss_val = model(x_val, y_val)

                # Normalize the loss over the number of validation steps
                validation_loss_total += loss_val.detach() / num_val_steps

        # Log the validation loss to the console and to the log file
        val_loss_value = validation_loss_total.item()
        print(f"Step {step}: Validation Loss: {val_loss_value:.4f}")
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Step {step}: val_loss {val_loss_value:.4f}\n")

        # ---------------------------------------------------------------------
        # 2b. Model Checkpointing
        # ---------------------------------------------------------------------
        # Save model checkpoints every 5000 steps (or on the last step)
        if step > 0 and (step % 5000 == 0 or is_last_step):
            checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
            checkpoint = {
                'model_state_dict': raw_model.state_dict(),
                'config': raw_model.config,
                'step': step,
                'val_loss': val_loss_value,
                # Optionally add optimizer state, RNG seeds, etc. for precise resumption.
            }
            torch.save(checkpoint, checkpoint_path)

    # --- (Place here your training update code for a single step) ---
    # For example:
    # model.train()
    # x_train, y_train = train_loader.next_batch()
    # x_train, y_train = x_train.to(device), y_train.to(device)
    # with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
    #     logits_train, loss_train = model(x_train, y_train)
    # loss_train.backward()
    # optimizer.step()
    # optimizer.zero_grad()
    # -------------------------------------------------------------------------

    # Optionally print the step duration (if needed)
    elapsed = time.time() - start_time
    print(f"Step {step} took {elapsed:.2f} sec")



# =============================================================================
# 1. Logging Setup
# =============================================================================
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "log.txt")
# Clear any previous log file
with open(log_file_path, "w") as f:
    pass

# =============================================================================
# 2. HellaSwag Evaluation
# =============================================================================
# Evaluate on the HellaSwag examples periodically (only if not using torch.compile)
if (step % 250 == 0 or is_last_step) and (not use_compile):
    correct_count = 0
    total_examples = 0

    # Iterate over HellaSwag examples from the validation set.
    # Note: `iterate_examples` is assumed to be defined elsewhere.
    for i, example in enumerate(iterate_examples("val")):
        # Render the example into token tensor, attention mask and label.
        # Note: `render_example` is assumed to be defined elsewhere.
        _, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # Get the logits from the model.
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            # Determine the most likely prediction using a helper function.
            # `get_most_likely_row` is assumed to be defined elsewhere.
            prediction = get_most_likely_completion(tokens, mask, logits)

        total_examples += 1
        correct_count += int(prediction == label)

    accuracy = correct_count / total_examples
    print(f"HellaSwag accuracy: {correct_count}/{total_examples} = {accuracy:.4f}")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{step} hella {accuracy:.4f}\n")

# =============================================================================
# 3. Sample Generation from the Model
# =============================================================================
# Periodically generate text samples (except at step 0, where output may be noisy)
if ((step > 0 and step % 250 == 0) or is_last_step) and (not use_compile):
    model.eval()
    num_return_sequences = 4
    max_sample_length = 32

    # Encode a prompt using the provided tokenizer (tiktoken)
    prompt_tokens = enc.encode("Hello, I'm a language model,")
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
    # Create a batch by repeating the prompt.
    input_sequences = prompt_tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    input_sequences = input_sequences.to(device)

    sample_rng = torch.Generator(device=device)
    sample_rng.manual_seed(42)  # Seed for reproducibility on a single device

    # Autoregressively generate tokens until reaching the maximum sample length.
    while input_sequences.size(1) < max_sample_length:
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(input_sequences)
            # Get the logits for the last token only.
            last_logits = logits[:, -1, :]  # shape: (batch, vocab_size)
            probs = F.softmax(last_logits, dim=-1)
            # Sample from the top-50 tokens (as per HuggingFace defaults)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            sampled_idx = torch.multinomial(topk_probs, 1, generator=sample_rng)
            next_token = torch.gather(topk_indices, -1, sampled_idx)
            input_sequences = torch.cat((input_sequences, next_token), dim=1)

    # Decode and print each generated sequence.
    for i in range(num_return_sequences):
        tokens_generated = input_sequences[i, :max_sample_length].tolist()
        decoded_text = enc.decode(tokens_generated)
        print(f"Sample {i}: {decoded_text}")

# =============================================================================
# 4. Training Step
# =============================================================================
model.train()
optimizer.zero_grad()
loss_accumulator = 0.0

for micro_step in range(grad_accum_steps):
    x_batch, y_batch = train_loader.next_batch()
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        logits, loss = model(x_batch, y_batch)
    # Adjust loss for gradient accumulation
    loss = loss / grad_accum_steps
    loss_accumulator += loss.detach()
    loss.backward()

# Clip gradients to prevent exploding gradients
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# Update learning rate for current step
current_lr = get_lr(step)
for param_group in optimizer.param_groups:
    param_group['lr'] = current_lr

optimizer.step()
if device_type == "cuda":
    torch.cuda.synchronize()  # Ensure GPU computations are complete

# Timing and performance logging
step_end_time = time.time()
elapsed_time = step_end_time - start_time  # seconds elapsed this step
# Calculate processed tokens: micro_batch_size * seq_length * grad_accum_steps
tokens_processed = train_loader.batch_size * train_loader.seq_length * grad_accum_steps
tokens_per_sec = tokens_processed / elapsed_time

print(f"Step {step:5d} | Loss: {loss_accumulator.item():.6f} | LR: {current_lr:.4e} | "
      f"Grad Norm: {grad_norm:.4f} | Time: {elapsed_time*1000:.2f} ms | Tokens/sec: {tokens_per_sec:.2f}")

with open(log_file_path, "a") as log_file:
    log_file.write(f"{step} train {loss_accumulator.item():.6f}\n")
