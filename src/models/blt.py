"""
Byte Latent Transformer (BLT) Local Patch Modules.
Implements Dynamic Entropy-Based Byte Local Encoder and Local Decoder modules
for dynamically segmenting raw byte sequences into variable-length latent patch representations
and unpacking them back to byte-level predictions (Meta AI, 2024 / BLT Architecture).
"""

import math
from typing import Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteEntropyEstimator(nn.Module):
    """
    Lightweight Byte Entropy Estimator for Dynamic BLT Patching.
    
    Computes conditional next-byte Shannon entropy:
        H(t) = - sum_{w} P(w | b_t) * log2(P(w | b_t))
    using an empirical byte transition matrix with Laplace smoothing.
    
    Args:
        vocab_size (int): Byte vocabulary size (260 = 256 bytes + 4 special tokens). Default: 260.
        smoothing (float): Laplace smoothing parameter for empirical transitions. Default: 1.0.
    """
    def __init__(self, vocab_size: int = 260, smoothing: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        
        # Buffer holding precomputed Shannon entropy for each byte token ID (0..vocab_size-1)
        # Default initialized to uniform entropy: log2(vocab_size)
        uniform_entropy = math.log2(vocab_size)
        self.register_buffer(
            "byte_entropy_table",
            torch.full((vocab_size,), fill_value=uniform_entropy, dtype=torch.float32)
        )
        # 2D transition counts buffer (vocab_size x vocab_size)
        self.register_buffer(
            "transition_counts",
            torch.zeros((vocab_size, vocab_size), dtype=torch.float32)
        )

    @torch.no_grad()
    def fit_sequences(self, sequences: List[List[int]]):
        """
        Fits empirical byte transition statistics over a corpus of token ID sequences
        and updates the per-byte entropy table.
        """
        counts = torch.zeros((self.vocab_size, self.vocab_size), dtype=torch.float32)
        for seq in sequences:
            if len(seq) < 2:
                continue
            for u, v in zip(seq[:-1], seq[1:]):
                if 0 <= u < self.vocab_size and 0 <= v < self.vocab_size:
                    counts[u, v] += 1.0
                    
        self.transition_counts.copy_(counts)
        self._update_entropy_table()

    @torch.no_grad()
    def _update_entropy_table(self):
        """Computes Shannon entropy H(v) for every byte token from transition counts."""
        # Add Laplace smoothing
        smoothed = self.transition_counts + self.smoothing
        probs = smoothed / smoothed.sum(dim=-1, keepdim=True)
        # H(u) = - sum_v p(v|u) * log2(p(v|u))
        entropy = -(probs * torch.log2(probs)).sum(dim=-1)
        
        # Special tokens (PAD=0, SOS=1, EOS=2, UNK=3) get low entropy to trigger boundary
        entropy[0] = 0.0
        entropy[1] = 0.0
        entropy[2] = 0.0
        entropy[3] = 0.0
        
        self.byte_entropy_table.copy_(entropy)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Looks up next-byte entropy for each token in byte_ids.
        
        Args:
            byte_ids: Tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Entropy values of shape (batch_size, seq_len)
        """
        # Clamp IDs safely to vocab range
        clamped_ids = byte_ids.clamp(0, self.vocab_size - 1)
        return self.byte_entropy_table[clamped_ids]


class ByteLocalEncoder(nn.Module):
    """
    Dynamic Entropy-Based Byte Local Encoder for Token-Free Transformer (BLT).
    
    Converts a raw sequence of byte IDs into latent patch representations using
    dynamic entropy-based boundary segmentation (Meta AI, 2024).
    
    Workflow:
    1. Byte Embedding Layer: byte_ids -> (batch, byte_len, d_byte)
    2. Local Feature Extraction: 1D Convolution over byte sequence
    3. Dynamic Entropy Patching: Dynamically segments bytes into variable-length patches
       [s_k, e_k) based on next-byte entropy H(t) exceeding threshold tau (bounded by [k_min, k_max]).
    4. Patch Pooling & Projection: Pools local byte features within each patch and projects -> d_model.
    5. Patch Mask Generation: Generates boolean attention mask for variable patch counts per batch.
    
    Args:
        vocab_size (int): Size of byte vocabulary (260). Default: 260.
        d_byte (int): Embedding dimension of raw bytes. Default: 64.
        d_model (int): Hidden dimension of global Transformer backbone. Default: 256.
        min_patch_len (int): Minimum bytes per patch (k_min). Default: 2.
        max_patch_len (int): Maximum bytes per patch (k_max). Default: 8.
        entropy_threshold (float): Entropy threshold tau in bits. Default: 4.0.
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        vocab_size: int = 260,
        d_byte: int = 64,
        d_model: int = 256,
        min_patch_len: int = 2,
        max_patch_len: int = 8,
        entropy_threshold: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        self.min_patch_len = max(1, min_patch_len)
        self.max_patch_len = max(self.min_patch_len, max_patch_len)
        self.entropy_threshold = entropy_threshold
        
        # 1. Byte-level learned embedding (256 bytes + 4 special tokens)
        self.byte_embedding = nn.Embedding(vocab_size, d_byte, padding_idx=0)
        
        # 2. Local 1D Convolution for intra-patch byte feature extraction
        self.conv = nn.Conv1d(
            in_channels=d_byte,
            out_channels=d_byte,
            kernel_size=3,
            padding=1
        )
        self.conv_norm = nn.LayerNorm(d_byte)
        
        # 3. Dynamic Entropy Estimator
        self.entropy_estimator = ByteEntropyEstimator(vocab_size=vocab_size)
        
        # 4. Patch projection: maps pooled local byte feature (d_byte) -> d_model
        self.patch_proj = nn.Sequential(
            nn.Linear(d_byte, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model)
        )
        self.norm = nn.LayerNorm(d_model)

    def _segment_patches(
        self,
        byte_ids_row: torch.Tensor,
        entropy_row: torch.Tensor,
        valid_len: int
    ) -> List[Tuple[int, int]]:
        """
        Dynamically segments a single byte sequence into patch intervals [start, end).
        A boundary is placed when:
          - Current patch length >= min_patch_len AND entropy >= threshold, OR
          - Current patch length reaches max_patch_len, OR
          - End of valid sequence is reached.
        """
        if valid_len <= 0:
            return [(0, 1)]
            
        patches = []
        start_idx = 0
        
        for t in range(valid_len):
            cur_len = t - start_idx + 1
            
            is_max_len = (cur_len >= self.max_patch_len)
            is_entropy_boundary = (cur_len >= self.min_patch_len and entropy_row[t].item() >= self.entropy_threshold)
            is_end = (t == valid_len - 1)
            
            if is_max_len or is_entropy_boundary or is_end:
                patches.append((start_idx, t + 1))
                start_idx = t + 1
                
        return patches

    def forward(
        self,
        byte_ids: torch.Tensor,
        byte_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass converting raw bytes to dynamic entropy patches.
        
        Args:
            byte_ids: Tensor of shape (batch_size, byte_seq_len) containing byte token IDs.
            byte_mask: Optional boolean tensor of shape (batch_size, byte_seq_len)
                       where True = valid token, False = padding.
                       
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - patch_embeds: Latent patches of shape (batch_size, max_patches, d_model)
                - patch_mask: Boolean tensor of shape (batch_size, max_patches)
        """
        batch_size, byte_seq_len = byte_ids.shape
        device = byte_ids.device
        
        # 1. Embed bytes: (batch, byte_seq_len, d_byte)
        byte_emb = self.byte_embedding(byte_ids)
        
        # 2. Local 1D Convolution over byte sequence
        conv_out = self.conv(byte_emb.transpose(1, 2)).transpose(1, 2)
        local_features = self.conv_norm(F.gelu(conv_out) + byte_emb)
        
        # 3. Calculate next-byte entropy: (batch, byte_seq_len)
        entropy_scores = self.entropy_estimator(byte_ids)
        
        # 4. Determine valid lengths for each item in the batch
        if byte_mask is not None:
            valid_lengths = byte_mask.sum(dim=1).tolist()
        else:
            valid_lengths = [byte_seq_len] * batch_size
            
        # 5. Segment each sample dynamically into patches and mean-pool
        batch_patch_reps: List[torch.Tensor] = []
        max_patches = 1
        
        for i in range(batch_size):
            v_len = int(valid_lengths[i]) if valid_lengths[i] > 0 else byte_seq_len
            patch_intervals = self._segment_patches(byte_ids[i], entropy_scores[i], v_len)
            max_patches = max(max_patches, len(patch_intervals))
            
            sample_patches = []
            for s, e in patch_intervals:
                # Mean pool byte features across the dynamically sized patch [s:e]
                patch_slice = local_features[i, s:e, :]  # (patch_len, d_byte)
                pooled = patch_slice.mean(dim=0)          # (d_byte,)
                sample_patches.append(pooled)
                
            sample_tensor = torch.stack(sample_patches, dim=0)  # (num_patches_i, d_byte)
            batch_patch_reps.append(sample_tensor)
            
        # 6. Collate into padded batch tensor: (batch_size, max_patches, d_byte)
        padded_patches = torch.zeros(
            batch_size, max_patches, self.d_byte,
            dtype=local_features.dtype, device=device
        )
        patch_mask = torch.zeros(
            batch_size, max_patches,
            dtype=torch.bool, device=device
        )
        
        for i, rep in enumerate(batch_patch_reps):
            n_p = rep.size(0)
            padded_patches[i, :n_p, :] = rep
            patch_mask[i, :n_p] = True
            
        # 7. Project patches to global Transformer dimension d_model: (batch, max_patches, d_model)
        patch_embeds = self.norm(self.patch_proj(padded_patches))
        
        return patch_embeds, patch_mask


class CausalConv1d(nn.Module):
    """
    1D Causal Convolution for Autoregressive Decoder Layers.
    Pads only on the left (t - kernel_size + 1 ... t), ensuring position t
    never receives information from future positions (t + 1).
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, channels, seq_len)
        # Pad only on the left: (pad_left, pad_right) = (self.pad, 0)
        x_padded = F.pad(x, (self.pad, 0))
        return self.conv(x_padded)


class ByteLocalDecoder(nn.Module):
    """
    Local Byte Decoder for Token-Free Transformer (BLT).
    
    Unpacks latent representations from the global Transformer backbone
    into byte sequences and predicts fine-grained byte logits autoregressively.
    
    Architecture:
    1. Linear Projection: maps (batch, tgt_len, d_model) -> (batch, tgt_len, d_byte)
    2. Strictly Causal Refinement: CausalConv1D across unpacked byte representations
    3. Byte Prediction Head: Linear layer mapping d_byte -> vocab_size (260)
    
    Args:
        vocab_size (int): Size of byte vocabulary (260). Default: 260.
        d_byte (int): Embedding dimension of bytes. Default: 64.
        d_model (int): Hidden dimension of global Transformer backbone. Default: 256.
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        vocab_size: int = 260,
        d_byte: int = 64,
        d_model: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_byte = d_byte
        self.d_model = d_model
        
        # 1. Project latent representation: d_model -> d_byte
        self.unpack_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_byte)
        )
        
        # 2. Strictly causal local refinement over byte sequence (no future token leakage)
        self.refine_conv = CausalConv1d(
            in_channels=d_byte,
            out_channels=d_byte,
            kernel_size=3
        )
        self.refine_norm = nn.LayerNorm(d_byte)
        
        # 3. Final byte classification head: d_byte -> vocab_size
        self.lm_head = nn.Linear(d_byte, vocab_size)

    def forward(
        self,
        decoder_latents: torch.Tensor,
        target_byte_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass unpacking latent representations to byte-level logits.
        
        Args:
            decoder_latents: Tensor of shape (batch_size, seq_len, d_model)
            target_byte_len: Optional exact byte sequence length to slice output to.
            
        Returns:
            torch.Tensor: Byte-level logits of shape (batch_size, seq_len, vocab_size)
        """
        # 1. Project latent: (batch, seq_len, d_byte)
        byte_features = self.unpack_proj(decoder_latents)
        
        # 2. Refine byte representations via Causal 1D Conv (permute for Conv1D: (batch, d_byte, seq_len))
        conv_out = self.refine_conv(byte_features.transpose(1, 2)).transpose(1, 2)
        refined_bytes = self.refine_norm(F.gelu(conv_out) + byte_features)
        
        # 3. Predict logits over the byte vocabulary: (batch, seq_len, vocab_size)
        logits = self.lm_head(refined_bytes)
        
        # 4. Trim to target_byte_len if specified
        if target_byte_len is not None and logits.size(1) > target_byte_len:
            logits = logits[:, :target_byte_len, :]
            
        return logits.contiguous()
