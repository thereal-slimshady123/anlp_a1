"""
Custom Normalization Layers for Transformer Architecture.
Implements LayerNorm and RMSNorm from scratch using fundamental PyTorch tensor operations.
"""

import torch
import torch.nn as nn
from typing import Union, Tuple, List


class LayerNorm(nn.Module):
    """
    Custom Layer Normalization (Ba et al., 2016).
    
    Normalizes activations across the channel/feature dimension:
        y = ((x - mean) / sqrt(var + eps)) * gamma + beta
        
    Args:
        d_model (int): Feature dimension (embedding size).
        eps (float): Small epsilon for numerical stability. Default: 1e-5.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable scale (gamma) and shift (beta) parameters
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for LayerNorm.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (*, d_model)
            
        Returns:
            Normalized tensor of same shape as x.
        """
        # Compute mean across the last dimension (d_model) -> shape: (*, 1)
        mean = x.mean(dim=-1, keepdim=True)
        
        # Compute variance across the last dimension without Bessel's correction (unbiased=False)
        # var = E[(x - mean)^2]
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Scale and shift with learnable parameters
        return x_norm * self.gamma + self.beta
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (Zhang & Sennrich, 2019).
    Used in modern LLM architectures (e.g., LLaMA, Mistral, Gemma).
    
    Normalizes activations using only the root mean square of feature values:
        RMS(x) = sqrt(mean(x^2) + eps)
        y = (x / RMS(x)) * gamma
        
    RMSNorm enforces scaling invariance without calculating or centering around the mean,
    reducing computational overhead by ~10-50% compared to standard LayerNorm.
    
    Args:
        d_model (int): Feature dimension (embedding size).
        eps (float): Small epsilon for numerical stability. Default: 1e-5.
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        # Learnable scale parameter (gamma). Note: RMSNorm does NOT use a bias parameter (beta).
        self.gamma = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RMSNorm.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (*, d_model)
            
        Returns:
            Normalized tensor of same shape as x.
        """
        # Calculate root mean square: sqrt( (1/d) * sum(x_i^2) + eps )
        # mean(x^2) computed along the last dimension
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x_norm = x / rms
        return x_norm * self.gamma
    
    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, eps={self.eps}"
