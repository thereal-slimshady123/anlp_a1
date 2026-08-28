"""
Custom Sequence-to-Sequence Transformer Framework from Scratch.
Implements Pre-LN Transformer Blocks, Encoder, Decoder, and unified Seq2SeqTransformer
supporting configurations C1, C2, C3, C4, and C5 (BLT).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union

from src.models.norm import LayerNorm, RMSNorm
from src.models.positional import (
    SinusoidalPositionalEncoding,
    RotaryPositionEmbedding,
    apply_rotary_pos_emb
)
from src.models.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    GroupedQueryAttention
)
from src.models.blt import ByteLocalEncoder, ByteLocalDecoder


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network (FFN).
    FFN(x) = GELU(x W_1 + b_1) W_2 + b_2
    
    Args:
        d_model (int): Hidden dimension.
        d_ff (int): Inner dimension of feedforward layer (typically 4 * d_model).
        dropout (float): Dropout probability. Default: 0.1.
        activation (str): Activation function ('gelu' or 'relu'). Default: 'gelu'.
    """
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation.lower() == 'gelu' else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.act(self.w1(x))))


class TransformerEncoderLayer(nn.Module):
    """
    Pre-LayerNorm Transformer Encoder Layer.
    
    Architecture (Pre-LN):
        x = x + Dropout( SelfAttention( Norm1(x) ) )
        x = x + Dropout( FFN( Norm2(x) ) )
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_type: str = 'mha',
        num_kv_heads: Optional[int] = None,
        norm_type: str = 'layernorm',
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.use_rope = use_rope
        self.attention_type = attention_type.lower()
        
        # 1. Normalization layers (LayerNorm vs RMSNorm)
        if norm_type.lower() == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            
        # 2. Self-Attention (MHA vs GQA)
        if self.attention_type == 'gqa':
            num_kv = num_kv_heads if num_kv_heads is not None else num_heads // 2
            self.self_attn = GroupedQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv,
                dropout=dropout
            )
        else:
            self.self_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            
        # 3. FeedForward
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        # Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(
            q=norm_x,
            k=norm_x,
            v=norm_x,
            mask=mask,
            rotary_pos_emb=rotary_pos_emb if self.use_rope else None
        )
        x = x + self.dropout(attn_out)
        
        # Pre-LN FeedForward
        norm_x = self.norm2(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Pre-LayerNorm Transformer Decoder Layer.
    
    Architecture (Pre-LN):
        x = x + Dropout( CausalSelfAttention( Norm1(x) ) )
        x = x + Dropout( CrossAttention( Norm2(x), Norm_mem(memory) ) )
        x = x + Dropout( FFN( Norm3(x) ) )
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_type: str = 'mha',
        num_kv_heads: Optional[int] = None,
        norm_type: str = 'layernorm',
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.use_rope = use_rope
        self.attention_type = attention_type.lower()
        
        # Normalization layers
        if norm_type.lower() == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
            self.norm_mem = RMSNorm(d_model)
        else:
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            self.norm3 = LayerNorm(d_model)
            self.norm_mem = LayerNorm(d_model)
            
        # Self-Attention (Causal)
        if self.attention_type == 'gqa':
            num_kv = num_kv_heads if num_kv_heads is not None else num_heads // 2
            self.self_attn = GroupedQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv,
                dropout=dropout
            )
            # Cross attention typically shares MHA/GQA module
            self.cross_attn = GroupedQueryAttention(
                d_model=d_model,
                num_heads=num_heads,
                num_kv_heads=num_kv,
                dropout=dropout
            )
        else:
            self.self_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            self.cross_attn = MultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout
            )
            
        self.ffn = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        # 1. Pre-LN Causal Self-Attention
        norm_x = self.norm1(x)
        self_out, _ = self.self_attn(
            q=norm_x,
            k=norm_x,
            v=norm_x,
            mask=tgt_mask,
            rotary_pos_emb=rotary_pos_emb if self.use_rope else None
        )
        x = x + self.dropout(self_out)
        
        # 2. Pre-LN Cross-Attention (Query = Decoder, Key/Value = Encoder Memory)
        norm_x = self.norm2(x)
        norm_mem = self.norm_mem(memory)
        cross_out, _ = self.cross_attn(
            q=norm_x,
            k=norm_mem,
            v=norm_mem,
            mask=memory_mask,
            rotary_pos_emb=None  # RoPE relative indexing usually across self-attention sequences
        )
        x = x + self.dropout(cross_out)
        
        # 3. Pre-LN FeedForward
        norm_x = self.norm3(x)
        ffn_out = self.ffn(norm_x)
        x = x + self.dropout(ffn_out)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of a stack of Pre-LN TransformerEncoderLayers.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_type: str = 'mha',
        num_kv_heads: Optional[int] = None,
        norm_type: str = 'layernorm',
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attention_type=attention_type,
                num_kv_heads=num_kv_heads,
                norm_type=norm_type,
                dropout=dropout,
                use_rope=use_rope
            )
            for _ in range(num_layers)
        ])
        
        if norm_type.lower() == 'rmsnorm':
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask, rotary_pos_emb=rotary_pos_emb)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of a stack of Pre-LN TransformerDecoderLayers.
    """
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        attention_type: str = 'mha',
        num_kv_heads: Optional[int] = None,
        norm_type: str = 'layernorm',
        dropout: float = 0.1,
        use_rope: bool = False
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                attention_type=attention_type,
                num_kv_heads=num_kv_heads,
                norm_type=norm_type,
                dropout=dropout,
                use_rope=use_rope
            )
            for _ in range(num_layers)
        ])
        
        if norm_type.lower() == 'rmsnorm':
            self.final_norm = RMSNorm(d_model)
        else:
            self.final_norm = LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(
                x,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                rotary_pos_emb=rotary_pos_emb
            )
        return self.final_norm(x)


class Seq2SeqTransformer(nn.Module):
    """
    Unified Sequence-to-Sequence Transformer Framework from Scratch.
    
    Supports the 5 target ablation configurations:
      - C1 (Base): Sinusoidal PE + MHA + LayerNorm + Subword Tokenizer
      - C2: RoPE + MHA + LayerNorm + Subword Tokenizer
      - C3: Sinusoidal PE + GQA + LayerNorm + Subword Tokenizer
      - C4: Sinusoidal PE + MHA + RMSNorm + Subword Tokenizer
      - C5 (BLT): Sinusoidal PE + MHA + LayerNorm + BLT (Byte Local Patching)
      
    Args:
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        d_model (int): Hidden dimension size. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_encoder_layers (int): Number of encoder layers. Default: 4.
        num_decoder_layers (int): Number of decoder layers. Default: 4.
        d_ff (int): Dimension of feedforward network. Default: 1024.
        pos_encoding (str): 'sinusoidal' or 'rope'. Default: 'sinusoidal'.
        attention_type (str): 'mha' or 'gqa'. Default: 'mha'.
        num_kv_heads (Optional[int]): Number of KV heads if attention_type is 'gqa'. Default: None (heads // 2).
        norm_type (str): 'layernorm' or 'rmsnorm'. Default: 'layernorm'.
        is_blt (bool): If True, activates Byte Latent Transformer architecture. Default: False.
        blt_patch_size (int): Patch size for BLT. Default: 4.
        blt_d_byte (int): Embedding size for raw bytes in BLT. Default: 64.
        max_len (int): Maximum sequence length supported. Default: 5000.
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        pos_encoding: str = 'sinusoidal',
        attention_type: str = 'mha',
        num_kv_heads: Optional[int] = None,
        norm_type: str = 'layernorm',
        is_blt: bool = False,
        blt_patch_size: int = 4,
        blt_d_byte: int = 64,
        max_len: int = 5000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.pos_encoding = pos_encoding.lower()
        self.attention_type = attention_type.lower()
        self.norm_type = norm_type.lower()
        self.is_blt = is_blt
        self.blt_patch_size = blt_patch_size
        self.max_len = max_len
        self.d_k = d_model // num_heads

        # 1. Token Embeddings / BLT Patch Modules
        if self.is_blt:
            # BLT: Token-free byte local encoder & decoder
            self.src_encoder = ByteLocalEncoder(
                vocab_size=src_vocab_size,
                d_byte=blt_d_byte,
                d_model=d_model,
                patch_size=blt_patch_size,
                dropout=dropout
            )
            # Decoder byte embedding for autoregressive target inputs
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
            self.blt_decoder = ByteLocalDecoder(
                vocab_size=tgt_vocab_size,
                d_byte=blt_d_byte,
                d_model=d_model,
                patch_size=blt_patch_size,
                dropout=dropout
            )
        else:
            self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
            self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
            self.output_projection = nn.Linear(d_model, tgt_vocab_size)
            
        # 2. Positional Embeddings
        self.use_rope = (self.pos_encoding == 'rope')
        if self.pos_encoding == 'sinusoidal':
            self.src_pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
            self.tgt_pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout)
            self.rope = None
        elif self.pos_encoding == 'rope':
            self.src_pos_enc = nn.Dropout(dropout)
            self.tgt_pos_enc = nn.Dropout(dropout)
            self.rope = RotaryPositionEmbedding(d_head=self.d_k, max_len=max_len)
        else:
            raise ValueError(f"Unsupported pos_encoding: {pos_encoding}")
            
        # 3. Transformer Encoder Backbone
        self.encoder = TransformerEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            attention_type=attention_type,
            num_kv_heads=num_kv_heads,
            norm_type=norm_type,
            dropout=dropout,
            use_rope=self.use_rope
        )
        
        # 4. Transformer Decoder Backbone
        self.decoder = TransformerDecoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            attention_type=attention_type,
            num_kv_heads=num_kv_heads,
            norm_type=norm_type,
            dropout=dropout,
            use_rope=self.use_rope
        )

    @classmethod
    def from_config_name(
        cls,
        config_name: str,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        d_ff: int = 1024,
        **kwargs
    ) -> 'Seq2SeqTransformer':
        """
        Factory method to construct model corresponding to C1 through C5:
        - C1 (Base): Sinusoidal PE + MHA + LayerNorm + Subword
        - C2: RoPE + MHA + LayerNorm + Subword
        - C3: Sinusoidal PE + GQA + LayerNorm + Subword
        - C4: Sinusoidal PE + MHA + RMSNorm + Subword
        - C5 (BLT): Sinusoidal PE + MHA + LayerNorm + BLT (Byte Local Patching)
        """
        cfg = config_name.upper()
        if cfg == 'C1':
            return cls(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=d_ff,
                pos_encoding='sinusoidal',
                attention_type='mha',
                norm_type='layernorm',
                is_blt=False,
                **kwargs
            )
        elif cfg == 'C2':
            return cls(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=d_ff,
                pos_encoding='rope',
                attention_type='mha',
                norm_type='layernorm',
                is_blt=False,
                **kwargs
            )
        elif cfg == 'C3':
            num_kv = kwargs.pop('num_kv_heads', max(1, num_heads // 2))
            return cls(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=d_ff,
                pos_encoding='sinusoidal',
                attention_type='gqa',
                num_kv_heads=num_kv,
                norm_type='layernorm',
                is_blt=False,
                **kwargs
            )
        elif cfg == 'C4':
            return cls(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=d_ff,
                pos_encoding='sinusoidal',
                attention_type='mha',
                norm_type='rmsnorm',
                is_blt=False,
                **kwargs
            )
        elif cfg == 'C5':
            return cls(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=d_model,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_ff=d_ff,
                pos_encoding='sinusoidal',
                attention_type='mha',
                norm_type='layernorm',
                is_blt=True,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown config name: {config_name}. Must be one of C1, C2, C3, C4, C5.")

    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encodes source sequence to latent memory representations.
        
        Args:
            src: (batch, src_seq_len)
            src_mask: (batch, src_seq_len) or broadcastable mask
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: (memory, effective_src_mask)
        """
        if self.is_blt:
            # BLT: Patch-level local encoding
            src_emb, eff_mask = self.src_encoder(src, byte_mask=src_mask)
            # Add sinusoidal positional encoding on patch level
            src_emb = self.src_pos_enc(src_emb)
        else:
            # Subword embedding scaled by sqrt(d_model) (standard Transformer)
            src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
            src_emb = self.src_pos_enc(src_emb)
            eff_mask = src_mask

        # Prepare 4D attention mask from 2D key padding mask if provided
        attn_mask = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                # (batch, 1, 1, seq_len)
                attn_mask = eff_mask.unsqueeze(1).unsqueeze(2)
            else:
                attn_mask = eff_mask

        # Compute RoPE embeddings if enabled
        rotary_pos = self.rope(src_emb.size(1), device=src.device) if self.use_rope else None
        
        memory = self.encoder(src_emb, mask=attn_mask, rotary_pos_emb=rotary_pos)
        return memory, eff_mask

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Decodes target sequence conditioned on encoder memory.
        
        Args:
            tgt: (batch, tgt_seq_len)
            memory: (batch, mem_seq_len, d_model)
            tgt_mask: (batch, 1, tgt_seq_len, tgt_seq_len)
            memory_mask: (batch, 1, 1, mem_seq_len)
        """
        batch_size, tgt_seq_len = tgt.shape
        
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_pos_enc(tgt_emb)
        
        rotary_pos = self.rope(tgt_seq_len, device=tgt.device) if self.use_rope else None
        
        dec_out = self.decoder(
            tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            rotary_pos_emb=rotary_pos
        )
        return dec_out

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            src: (batch, src_len)
            tgt: (batch, tgt_len)
            src_mask: (batch, src_len) boolean padding mask
            tgt_mask: (batch, 1, tgt_len, tgt_len) causal + padding mask
            
        Returns:
            logits: (batch, tgt_len, tgt_vocab_size)
        """
        memory, eff_src_mask = self.encode(src, src_mask=src_mask)
        
        mem_mask_4d = None
        if eff_src_mask is not None:
            if eff_src_mask.dim() == 2:
                mem_mask_4d = eff_src_mask.unsqueeze(1).unsqueeze(2)
            else:
                mem_mask_4d = eff_src_mask
                
        dec_out = self.decode(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=mem_mask_4d
        )
        
        if self.is_blt:
            # BLT: Project decoder latents through local byte decoder
            logits = self.blt_decoder(dec_out, target_byte_len=tgt.size(1))
        else:
            # Standard projection head
            logits = self.output_projection(dec_out)
            
        return logits
