"""
Custom Positional Encoding Modules for Transformer Architecture.
Implements Sinusoidal Absolute Positional Encoding (Vaswani et al., 2017)
and Rotary Position Embedding (RoPE) (Su et al., 2021) from scratch.
"""

import math
import torch
import torch.nn as nn
from typing import Tuple, Optional


class SinusoidalPositionalEncoding(nn.Module):
    """
    Absolute Sinusoidal Positional Encoding (Vaswani et al., 2017).
    
    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    
    Args:
        d_model (int): Feature dimension of token embeddings. Must be even.
        max_len (int): Maximum sequence length supported. Default: 5000.
        dropout (float): Dropout probability applied after adding PE. Default: 0.1.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for sinusoidal positional encoding, got {d_model}")
            
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Precompute the positional encoding matrix: shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: shape (max_len, 1) -> [[0], [1], ..., [max_len-1]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Frequency scale factors: 10000^(2i / d_model) = exp(2i * -log(10000) / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        
        # Apply sine to even indices: 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices: 2i + 1
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # Register as non-trainable persistent buffer (saved in state_dict, moved with .to(device))
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass adding positional encoding to input embeddings.
        
        Args:
            x (torch.Tensor): Input token embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Positionally encoded embeddings of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.pe.size(1)}")
            
        # Add positional encoding up to current seq_len
        # self.pe[:, :seq_len, :] broadcasts across batch_size
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) (Su et al., 2021).
    
    Encodes relative position by rotating Query and Key vectors in 2D sub-planes.
    Given vector x at position m, for each 2D chunk [x_2i, x_2i+1]:
        [x'_2i  ] = [cos(m * theta_i)  -sin(m * theta_i)] [x_2i  ]
        [x'_2i+1]   [sin(m * theta_i)   cos(m * theta_i)] [x_2i+1]
        
    where theta_i = 10000^(-2i / d_head).
    
    Properties:
    - Inner product <R_m q, R_n k> depends only on the relative distance (m - n).
    - Preserves norm of queries and keys (orthogonal transformation).
    - Can be applied to varying sequence lengths on the fly.
    
    Args:
        d_head (int): Head dimension on which RoPE is applied. Must be even.
        max_len (int): Maximum sequence length to precompute cache. Default: 5000.
        base (float): Base for the frequency geometric progression. Default: 10000.0.
    """
    def __init__(self, d_head: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        if d_head % 2 != 0:
            raise ValueError(f"d_head must be even for RoPE, got {d_head}")
            
        self.d_head = d_head
        self.max_len = max_len
        self.base = base
        
        # Precompute inverse frequencies: theta_i = 1 / (base ^ (2i / d_head))
        # shape: (d_head // 2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin caches for fast indexing during forward passes
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        """Precomputes cos and sin tables up to seq_len."""
        # Positions: (seq_len,)
        t = torch.arange(seq_len, dtype=torch.float, device=self.inv_freq.device)
        # Outer product: (seq_len, d_head // 2)
        freqs = torch.outer(t, self.inv_freq)
        # Concatenate freqs to match full d_head: (seq_len, d_head)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin: shape (seq_len, d_head)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns cached (cos, sin) tensors sliced to seq_len.
        
        Args:
            seq_len (int): Length of the sequence.
            device (torch.device, optional): Device on which tensors should reside.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos, sin) each of shape (1, 1, seq_len, d_head)
        """
        if seq_len > self.cos_cached.size(0):
            self._build_cache(max(seq_len, self.max_len * 2))
            
        cos = self.cos_cached[:seq_len].to(device=device)
        sin = self.sin_cached[:seq_len].to(device=device)
        
        # Expand shape for broadcasting with (batch_size, num_heads, seq_len, d_head)
        return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotates half the hidden dimensions of input tensor x.
    Transforms [-x2, x1] chunk-wise for the RoPE formulation:
    [-x_{d/2 :}, x_{: d/2}]
    
    Args:
        x (torch.Tensor): Tensor of shape (..., d_head)
        
    Returns:
        torch.Tensor: Rotated tensor of shape (..., d_head)
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Position Embedding to tensor x (e.g., Q or K).
    
    Formula:
        x_rot = (x * cos) + (rotate_half(x) * sin)
        
    Args:
        x (torch.Tensor): Tensor of shape (batch_size, num_heads, seq_len, d_head)
                          or (batch_size, seq_len, num_heads, d_head)
        cos (torch.Tensor): Precomputed cosine tensor of broadcastable shape
        sin (torch.Tensor): Precomputed sine tensor of broadcastable shape
        
    Returns:
        torch.Tensor: Rotated tensor of same shape as x.
    """
    return (x * cos) + (rotate_half(x) * sin)
