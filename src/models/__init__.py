"""
Transformer Models Module.
Exports all low-level building blocks and the unified Seq2SeqTransformer architecture.
"""

from src.models.norm import LayerNorm, RMSNorm
from src.models.positional import (
    SinusoidalPositionalEncoding,
    RotaryPositionEmbedding,
    apply_rotary_pos_emb,
    rotate_half
)
from src.models.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    GroupedQueryAttention,
    repeat_kv
)
from src.models.blt import ByteLocalEncoder, ByteLocalDecoder
from src.models.transformer import (
    FeedForward,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Seq2SeqTransformer
)

__all__ = [
    'LayerNorm',
    'RMSNorm',
    'SinusoidalPositionalEncoding',
    'RotaryPositionEmbedding',
    'apply_rotary_pos_emb',
    'rotate_half',
    'ScaledDotProductAttention',
    'MultiHeadAttention',
    'GroupedQueryAttention',
    'repeat_kv',
    'ByteLocalEncoder',
    'ByteLocalDecoder',
    'FeedForward',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
    'TransformerEncoder',
    'TransformerDecoder',
    'Seq2SeqTransformer'
]
