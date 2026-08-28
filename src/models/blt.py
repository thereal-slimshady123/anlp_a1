"""
Byte Latent Transformer (BLT) Local Patch Modules.
Implements Token-Free Byte Local Encoder and Local Decoder modules
for dynamically grouping raw byte sequences into latent patch representations
and unpacking them back to byte-level predictions (Meta AI, 2024 / Token-Free Transformer).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ByteLocalEncoder(nn.Module):
    """
    Local Byte Encoder for Token-Free Transformer (BLT).
    
    Converts a raw sequence of byte IDs into latent patch representations.
    Groups non-overlapping or strided windows of bytes of size `patch_size`
    and projects them into the latent Transformer dimension `d_model`.
    
    Architecture:
    1. Byte Embedding Layer: byte_ids -> (batch, byte_len, d_byte)
    2. Local Feature Extraction: 1D Convolution / Linear projection over byte patches
    3. Patch Projection: Projects concatenated / pooled local features -> d_model
    4. Patch Padding Mask Generation: Derives patch-level attention masks from byte masks
    
    Args:
        vocab_size (int): Size of byte vocabulary (256 raw bytes + special tokens = 260). Default: 260.
        d_byte (int): Embedding dimension of raw bytes. Default: 64.
        d_model (int): Hidden dimension of global Transformer backbone. Default: 256.
        patch_size (int): Number of consecutive bytes per latent patch. Default: 4.
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        vocab_size: int = 260,
        d_byte: int = 64,
        d_model: int = 256,
        patch_size: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 1. Byte-level embedding
        self.byte_embedding = nn.Embedding(vocab_size, d_byte, padding_idx=0)
        
        # 2. Local 1D Convolution for intra-patch feature extraction
        self.conv = nn.Conv1d(
            in_channels=d_byte,
            out_channels=d_byte,
            kernel_size=3,
            padding=1
        )
        self.conv_norm = nn.LayerNorm(d_byte)
        
        # 3. Patch projection: maps (patch_size * d_byte) -> d_model
        self.patch_proj = nn.Sequential(
            nn.Linear(d_byte * patch_size, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass converting raw bytes to latent patches.
        
        Args:
            byte_ids: Tensor of shape (batch_size, byte_seq_len) containing byte token IDs.
            byte_mask: Optional boolean tensor of shape (batch_size, byte_seq_len)
                       where True = valid token, False = padding.
                       
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - patch_embeds: Latent patches of shape (batch_size, num_patches, d_model)
                - patch_mask: Boolean tensor of shape (batch_size, num_patches)
        """
        batch_size, byte_seq_len = byte_ids.shape
        
        # Pad byte_seq_len to be an exact multiple of patch_size if needed
        remainder = byte_seq_len % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            byte_ids = F.pad(byte_ids, (0, pad_len), value=0)
            if byte_mask is not None:
                byte_mask = F.pad(byte_mask, (0, pad_len), value=False)
            byte_seq_len = byte_ids.shape[1]
            
        num_patches = byte_seq_len // self.patch_size
        
        # 1. Embed bytes: (batch, byte_seq_len, d_byte)
        byte_emb = self.byte_embedding(byte_ids)
        
        # 2. Local 1D Convolution over byte sequence
        # Permute for Conv1D: (batch, d_byte, byte_seq_len)
        conv_out = self.conv(byte_emb.transpose(1, 2)).transpose(1, 2)
        conv_out = self.conv_norm(F.gelu(conv_out) + byte_emb)
        
        # 3. Reshape into non-overlapping patches: (batch, num_patches, patch_size * d_byte)
        patches_flat = conv_out.view(batch_size, num_patches, self.patch_size * self.d_byte)
        
        # 4. Project patches to global Transformer dimension d_model: (batch, num_patches, d_model)
        patch_embeds = self.norm(self.patch_proj(patches_flat))
        
        # 5. Derive patch-level mask: A patch is valid if AT LEAST ONE byte in it is valid
        patch_mask = None
        if byte_mask is not None:
            # Reshape mask: (batch, num_patches, patch_size) -> (batch, num_patches)
            patch_mask = byte_mask.view(batch_size, num_patches, self.patch_size).any(dim=-1)
            
        return patch_embeds, patch_mask


class ByteLocalDecoder(nn.Module):
    """
    Local Byte Decoder for Token-Free Transformer (BLT).
    
    Unpacks latent patch representations from the global Transformer backbone
    back into fine-grained byte sequences and predicts byte logits.
    
    Architecture:
    1. Patch Unpooling / Linear Expansion: maps (batch, num_patches, d_model)
       -> (batch, num_patches * patch_size, d_byte)
    2. Local refinement layer: Conv1D / Residual block across unpacked bytes
    3. Byte Prediction Head: Linear layer mapping d_byte -> vocab_size
    
    Args:
        vocab_size (int): Size of byte vocabulary (260). Default: 260.
        d_byte (int): Embedding dimension of bytes. Default: 64.
        d_model (int): Hidden dimension of global Transformer backbone. Default: 256.
        patch_size (int): Number of bytes represented by each latent patch. Default: 4.
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        vocab_size: int = 260,
        d_byte: int = 64,
        d_model: int = 256,
        patch_size: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.patch_size = patch_size
        
        # 1. Unpack patch representation into patch_size byte vectors:
        # maps d_model -> (patch_size * d_byte)
        self.unpack_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, patch_size * d_byte)
        )
        
        # 2. Local refinement over unpacked bytes
        self.refine_conv = nn.Conv1d(
            in_channels=d_byte,
            out_channels=d_byte,
            kernel_size=3,
            padding=1
        )
        self.refine_norm = nn.LayerNorm(d_byte)
        
        # 3. Final byte classification head: d_byte -> vocab_size
        self.lm_head = nn.Linear(d_byte, vocab_size)

    def forward(
        self,
        patch_latents: torch.Tensor,
        target_byte_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass unpacking latent patches to byte-level logits.
        
        Args:
            patch_latents: Tensor of shape (batch_size, num_patches, d_model)
            target_byte_len: Optional exact byte sequence length to slice/trim output to.
            
        Returns:
            torch.Tensor: Byte-level logits of shape (batch_size, byte_seq_len, vocab_size)
        """
        batch_size, num_patches, _ = patch_latents.shape
        
        # 1. Project latent patches: (batch, num_patches, patch_size * d_byte)
        unpacked_flat = self.unpack_proj(patch_latents)
        
        # 2. Reshape into byte sequence: (batch, num_patches * patch_size, d_byte)
        byte_features = unpacked_flat.view(batch_size, num_patches * self.patch_size, self.d_byte)
        
        # 3. Refine byte representations via 1D Conv
        conv_out = self.refine_conv(byte_features.transpose(1, 2)).transpose(1, 2)
        refined_bytes = self.refine_norm(F.gelu(conv_out) + byte_features)
        
        # 4. Predict logits over the byte vocabulary: (batch, byte_seq_len, vocab_size)
        logits = self.lm_head(refined_bytes)
        
        # 5. Trim to target_byte_len if specified
        if target_byte_len is not None and logits.size(1) > target_byte_len:
            logits = logits[:, :target_byte_len, :]
            
        return logits
