"""
Custom Attention Mechanisms for Transformer Architecture.
Implements ScaledDotProductAttention, MultiHeadAttention (MHA),
and GroupedQueryAttention (GQA) from scratch using fundamental PyTorch tensor operations.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention (Vaswani et al., 2017).
    
    Attention(Q, K, V) = softmax( (Q K^T) / sqrt(d_k) + mask ) V
    
    Args:
        dropout (float): Dropout probability applied to attention weights. Default: 0.1.
    """
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Scaled Dot-Product Attention.
        
        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            k: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            v: Value tensor of shape (batch_size, num_heads, seq_len_k, d_v)
            mask: Optional attention mask tensor broadcastable to
                  (batch_size, num_heads, seq_len_q, seq_len_k).
                  Can be a boolean mask (True = keep, False = mask out)
                  or additive float mask (0.0 = keep, -inf = mask out).
                  
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output context tensor: (batch_size, num_heads, seq_len_q, d_v)
                - Attention weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        d_k = q.size(-1)
        
        # Compute raw attention scores: (batch, heads, seq_len_q, seq_len_k)
        # q @ k.transpose(-2, -1) performs batched matrix multiplication
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dtype == torch.bool:
                # If boolean mask, fill False positions with a very large negative number
                # or fill True positions where mask is a masking condition
                # Standard convention: True indicates valid token (keep), False indicates masked token
                scores = scores.masked_fill(~mask, float('-inf'))
            else:
                # Additive mask (e.g. causal mask with 0.0 and -inf)
                scores = scores + mask
                
        # Softmax along the last dimension (over key sequence length)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Handle cases where an entire row is masked out (softmax produces NaNs)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        # Apply dropout to attention probabilities
        attn_weights_dropped = self.dropout(attn_weights)
        
        # Weighted sum of values: (batch, heads, seq_len_q, d_v)
        output = torch.matmul(attn_weights_dropped, v)
        
        return output, attn_weights


def repeat_kv(x: torch.Tensor, num_rep: int) -> torch.Tensor:
    """
    Repeats key/value heads along the head dimension for Grouped-Query Attention (GQA).
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_kv_heads, seq_len, d_k)
        num_rep (int): Number of repetitions per KV head (group size = num_q_heads // num_kv_heads)
        
    Returns:
        torch.Tensor: Repeated tensor of shape (batch_size, num_kv_heads * num_rep, seq_len, d_k)
    """
    if num_rep == 1:
        return x
        
    batch_size, num_kv_heads, seq_len, d_k = x.shape
    # Expand: (batch, num_kv_heads, 1, seq_len, d_k) -> (batch, num_kv_heads, num_rep, seq_len, d_k)
    # Then reshape to: (batch, num_kv_heads * num_rep, seq_len, d_k)
    return (
        x.unsqueeze(2)
        .expand(batch_size, num_kv_heads, num_rep, seq_len, d_k)
        .reshape(batch_size, num_kv_heads * num_rep, seq_len, d_k)
    )


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA) Layer (Vaswani et al., 2017).
    Used in Baseline (C1), RoPE (C2), RMSNorm (C4), and BLT (C5).
    
    Splits feature dimension d_model into num_heads parallel attention heads,
    computes scaled dot-product attention in parallel, and projects back to d_model.
    
    Args:
        d_model (int): Feature dimension (embedding size).
        num_heads (int): Number of attention heads. d_model must be divisible by num_heads.
        dropout (float): Dropout probability. Default: 0.1.
        bias (bool): Whether to include bias in linear projections. Default: True.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
            
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Query, Key, Value, and Output
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_k, d_model)
            mask: Optional mask broadcastable to (batch_size, 1, seq_len_q, seq_len_k)
            rotary_pos_emb: Optional tuple of (cos, sin) precomputed RoPE tensors
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output context tensor: (batch_size, seq_len_q, d_model)
                - Attention weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = q.shape
        seq_len_k = k.size(1)
        
        # 1. Project inputs: (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)
        
        # 2. Reshape into heads: (batch, num_heads, seq_len, d_k)
        q_heads = q_proj.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = k_proj.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        v_heads = v_proj.view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply RoPE if provided (rotates Q and K heads)
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            # Lazy import helper to avoid circular dependencies
            from src.models.positional import apply_rotary_pos_emb
            q_heads = apply_rotary_pos_emb(q_heads, cos, sin)
            k_heads = apply_rotary_pos_emb(k_heads, cos, sin)
            
        # 4. Scaled Dot-Product Attention across all heads
        attn_out, attn_weights = self.attention(q_heads, k_heads, v_heads, mask=mask)
        
        # 5. Concatenate heads back: (batch, seq_len_q, num_heads, d_k) -> (batch, seq_len_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 6. Final linear output projection
        output = self.out_proj(attn_out)
        
        return output, attn_weights


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) Layer (Ainslie et al., 2023).
    Used in Configuration C3.
    
    Interpolates between Multi-Head Attention (MHA, where num_kv_heads == num_heads)
    and Multi-Query Attention (MQA, where num_kv_heads == 1).
    Groups query heads into subsets that share a single key/value head, significantly
    reducing KV cache memory and memory bandwidth bottlenecks during inference.
    
    Args:
        d_model (int): Feature dimension (embedding size).
        num_heads (int): Number of Query attention heads.
        num_kv_heads (int): Number of Key/Value attention heads. Must divide num_heads.
        dropout (float): Dropout probability. Default: 0.1.
        bias (bool): Whether to include bias in linear projections. Default: True.
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
            
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.d_k = d_model // num_heads
        
        # Projections: Q projects to num_heads * d_k (d_model), while K and V project to num_kv_heads * d_k
        self.q_proj = nn.Linear(d_model, self.num_heads * self.d_k, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=bias)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.d_k, bias=bias)
        self.out_proj = nn.Linear(self.num_heads * self.d_k, d_model, bias=bias)
        
        self.attention = ScaledDotProductAttention(dropout=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Grouped-Query Attention.
        
        Args:
            q: Query tensor of shape (batch_size, seq_len_q, d_model)
            k: Key tensor of shape (batch_size, seq_len_k, d_model)
            v: Value tensor of shape (batch_size, seq_len_k, d_model)
            mask: Optional mask broadcastable to (batch_size, 1, seq_len_q, seq_len_k)
            rotary_pos_emb: Optional tuple of (cos, sin) precomputed RoPE tensors
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output context tensor: (batch_size, seq_len_q, d_model)
                - Attention weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = q.shape
        seq_len_k = k.size(1)
        
        # 1. Project inputs:
        # q: (batch, seq_len_q, num_heads * d_k)
        # k, v: (batch, seq_len_k, num_kv_heads * d_k)
        q_proj = self.q_proj(q)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(v)
        
        # 2. Reshape into heads:
        # q_heads: (batch, num_heads, seq_len_q, d_k)
        # k_heads, v_heads: (batch, num_kv_heads, seq_len_k, d_k)
        q_heads = q_proj.view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        k_heads = k_proj.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        v_heads = v_proj.view(batch_size, seq_len_k, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # 3. Apply RoPE if provided
        if rotary_pos_emb is not None:
            cos, sin = rotary_pos_emb
            from src.models.positional import apply_rotary_pos_emb
            q_heads = apply_rotary_pos_emb(q_heads, cos, sin)
            k_heads = apply_rotary_pos_emb(k_heads, cos, sin)
            
        # 4. Repeat KV heads across query groups:
        # (batch, num_kv_heads, seq_len_k, d_k) -> (batch, num_heads, seq_len_k, d_k)
        k_expanded = repeat_kv(k_heads, self.group_size)
        v_expanded = repeat_kv(v_heads, self.group_size)
        
        # 5. Scaled Dot-Product Attention
        attn_out, attn_weights = self.attention(q_heads, k_expanded, v_expanded, mask=mask)
        
        # 6. Concatenate heads back to d_model: (batch, seq_len_q, d_model)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # 7. Final linear output projection
        output = self.out_proj(attn_out)
        
        return output, attn_weights
