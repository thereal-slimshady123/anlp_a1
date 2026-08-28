"""
Unit Tests for Custom Transformer Modules.
Tests LayerNorm, RMSNorm, Positional Encodings, Attention Mechanisms, BLT, and Full Model.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
from src.models.blt import (
    ByteLocalEncoder,
    ByteLocalDecoder
)


class TestNormModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 4
        self.seq_len = 16
        self.d_model = 64
        self.x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)

    def test_layernorm_shape_and_grad(self):
        ln = LayerNorm(d_model=self.d_model)
        out = ln(self.x)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        
        mean = out.mean(dim=-1)
        std = out.std(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-4))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(self.x.grad)
        self.assertIsNotNone(ln.gamma.grad)
        self.assertIsNotNone(ln.beta.grad)

    def test_rmsnorm_shape_and_grad(self):
        rmsn = RMSNorm(d_model=self.d_model)
        x_rms = self.x.clone().detach().requires_grad_(True)
        out = rmsn(x_rms)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        
        rms_val = torch.sqrt(torch.mean(out ** 2, dim=-1))
        self.assertTrue(torch.allclose(rms_val, torch.ones_like(rms_val), atol=1e-4))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x_rms.grad)
        self.assertIsNotNone(rmsn.gamma.grad)

    def test_layernorm_vs_rmsnorm_invariance(self):
        rmsn = RMSNorm(d_model=self.d_model)
        x_orig = torch.randn(2, 8, self.d_model)
        x_scaled = x_orig * 5.0
        out1 = rmsn(x_orig)
        out2 = rmsn(x_scaled)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))


class TestPositionalEncodingModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 16
        self.d_model = 64
        self.d_head = self.d_model // self.num_heads  # 16

    def test_sinusoidal_pe_shape_and_no_trainable_params(self):
        pe = SinusoidalPositionalEncoding(d_model=self.d_model, max_len=100, dropout=0.0)
        param_count = sum(p.numel() for p in pe.parameters() if p.requires_grad)
        self.assertEqual(param_count, 0)
        
        x = torch.randn(self.batch_size, self.seq_len, self.d_model, requires_grad=True)
        out = pe(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.d_model))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_rope_shape_and_relative_shift_invariance(self):
        rope = RotaryPositionEmbedding(d_head=self.d_head, max_len=128)
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_head, requires_grad=True)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.d_head, requires_grad=True)
        
        cos, sin = rope(seq_len=self.seq_len)
        self.assertEqual(cos.shape, (1, 1, self.seq_len, self.d_head))
        self.assertEqual(sin.shape, (1, 1, self.seq_len, self.d_head))
        
        q_rot = apply_rotary_pos_emb(q, cos, sin)
        k_rot = apply_rotary_pos_emb(k, cos, sin)
        
        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)
        
        loss = (q_rot * k_rot).sum()
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)

    def test_rope_mathematical_relative_property(self):
        rope = RotaryPositionEmbedding(d_head=self.d_head, max_len=128)
        cos_full, sin_full = rope(seq_len=64)
        
        q_vec = torch.randn(1, 1, 1, self.d_head)
        k_vec = torch.randn(1, 1, 1, self.d_head)
        
        m, n = 5, 2
        shift = 10
        
        q_m = apply_rotary_pos_emb(q_vec, cos_full[:, :, m:m+1, :], sin_full[:, :, m:m+1, :])
        k_n = apply_rotary_pos_emb(k_vec, cos_full[:, :, n:n+1, :], sin_full[:, :, n:n+1, :])
        score_1 = (q_m * k_n).sum()
        
        q_m_shifted = apply_rotary_pos_emb(q_vec, cos_full[:, :, (m+shift):(m+shift+1), :], sin_full[:, :, (m+shift):(m+shift+1), :])
        k_n_shifted = apply_rotary_pos_emb(k_vec, cos_full[:, :, (n+shift):(n+shift+1), :], sin_full[:, :, (n+shift):(n+shift+1), :])
        score_2 = (q_m_shifted * k_n_shifted).sum()
        
        self.assertTrue(torch.allclose(score_1, score_2, atol=1e-5))


class TestAttentionModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len_q = 8
        self.seq_len_kv = 12
        self.d_model = 64
        self.num_heads = 4
        self.num_kv_heads = 2
        self.d_k = self.d_model // self.num_heads

    def test_scaled_dot_product_attention_masking(self):
        attn = ScaledDotProductAttention(dropout=0.0)
        q = torch.randn(self.batch_size, self.num_heads, self.seq_len_q, self.d_k)
        k = torch.randn(self.batch_size, self.num_heads, self.seq_len_kv, self.d_k)
        v = torch.randn(self.batch_size, self.num_heads, self.seq_len_kv, self.d_k)
        
        mask = torch.ones(self.batch_size, 1, 1, self.seq_len_kv, dtype=torch.bool)
        mask[:, :, :, -4:] = False
        
        out, weights = attn(q, k, v, mask=mask)
        
        self.assertEqual(out.shape, (self.batch_size, self.num_heads, self.seq_len_q, self.d_k))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_len_q, self.seq_len_kv))
        
        masked_weights = weights[:, :, :, -4:]
        self.assertTrue(torch.allclose(masked_weights, torch.zeros_like(masked_weights)))
        
        row_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

    def test_multi_head_attention_shapes_and_cross_attention(self):
        mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout=0.0)
        
        x = torch.randn(self.batch_size, self.seq_len_q, self.d_model, requires_grad=True)
        out_self, _ = mha(x, x, x)
        self.assertEqual(out_self.shape, (self.batch_size, self.seq_len_q, self.d_model))
        
        memory = torch.randn(self.batch_size, self.seq_len_kv, self.d_model, requires_grad=True)
        out_cross, _ = mha(x, memory, memory)
        self.assertEqual(out_cross.shape, (self.batch_size, self.seq_len_q, self.d_model))
        
        loss = out_cross.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(memory.grad)
        self.assertIsNotNone(mha.q_proj.weight.grad)

    def test_grouped_query_attention_shapes_and_grad(self):
        gqa = GroupedQueryAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            dropout=0.0
        )
        
        q = torch.randn(self.batch_size, self.seq_len_q, self.d_model, requires_grad=True)
        kv = torch.randn(self.batch_size, self.seq_len_kv, self.d_model, requires_grad=True)
        
        out, weights = gqa(q, kv, kv)
        
        self.assertEqual(out.shape, (self.batch_size, self.seq_len_q, self.d_model))
        self.assertEqual(weights.shape, (self.batch_size, self.num_heads, self.seq_len_q, self.seq_len_kv))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(kv.grad)
        self.assertIsNotNone(gqa.k_proj.weight.grad)

    def test_mha_with_rope_integration(self):
        mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads, dropout=0.0)
        rope = RotaryPositionEmbedding(d_head=self.d_k, max_len=64)
        
        x = torch.randn(self.batch_size, self.seq_len_q, self.d_model, requires_grad=True)
        cos, sin = rope(seq_len=self.seq_len_q)
        
        out, _ = mha(x, x, x, rotary_pos_emb=(cos, sin))
        self.assertEqual(out.shape, (self.batch_size, self.seq_len_q, self.d_model))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)


class TestBLTModules(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.byte_seq_len = 32  # e.g., 32 bytes
        self.vocab_size = 260
        self.d_byte = 32
        self.d_model = 64
        self.patch_size = 4
        self.expected_num_patches = self.byte_seq_len // self.patch_size  # 8

    def test_byte_local_encoder_shape_and_mask(self):
        encoder = ByteLocalEncoder(
            vocab_size=self.vocab_size,
            d_byte=self.d_byte,
            d_model=self.d_model,
            patch_size=self.patch_size
        )
        
        byte_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.byte_seq_len))
        byte_mask = torch.ones(self.batch_size, self.byte_seq_len, dtype=torch.bool)
        # Mask out the last 8 bytes
        byte_mask[:, -8:] = False
        
        patch_embeds, patch_mask = encoder(byte_ids, byte_mask)
        
        # Check shape: (batch, num_patches, d_model)
        self.assertEqual(patch_embeds.shape, (self.batch_size, self.expected_num_patches, self.d_model))
        # Check patch mask shape: (batch, num_patches)
        self.assertEqual(patch_mask.shape, (self.batch_size, self.expected_num_patches))
        # Last 2 patches (8 bytes / 4) should be False
        self.assertFalse(patch_mask[:, -2:].any().item())
        self.assertTrue(patch_mask[:, :-2].all().item())

    def test_byte_local_decoder_shape_and_grad(self):
        decoder = ByteLocalDecoder(
            vocab_size=self.vocab_size,
            d_byte=self.d_byte,
            d_model=self.d_model,
            patch_size=self.patch_size
        )
        
        patch_latents = torch.randn(
            self.batch_size,
            self.expected_num_patches,
            self.d_model,
            requires_grad=True
        )
        
        logits = decoder(patch_latents, target_byte_len=self.byte_seq_len)
        
        # Check shape: (batch, byte_seq_len, vocab_size)
        self.assertEqual(logits.shape, (self.batch_size, self.byte_seq_len, self.vocab_size))
        
        # Test backward pass
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(patch_latents.grad)
        self.assertIsNotNone(decoder.lm_head.weight.grad)

    def test_blt_end_to_end_gradient_flow(self):
        encoder = ByteLocalEncoder(
            vocab_size=self.vocab_size,
            d_byte=self.d_byte,
            d_model=self.d_model,
            patch_size=self.patch_size
        )
        decoder = ByteLocalDecoder(
            vocab_size=self.vocab_size,
            d_byte=self.d_byte,
            d_model=self.d_model,
            patch_size=self.patch_size
        )
        
        byte_ids = torch.randint(1, self.vocab_size, (self.batch_size, self.byte_seq_len))
        patch_latents, _ = encoder(byte_ids)
        logits = decoder(patch_latents, target_byte_len=self.byte_seq_len)
        
        target = torch.randint(0, self.vocab_size, (self.batch_size, self.byte_seq_len))
        loss = nn.CrossEntropyLoss()(logits.view(-1, self.vocab_size), target.view(-1))
        loss.backward()
        
        self.assertIsNotNone(encoder.byte_embedding.weight.grad)
        self.assertIsNotNone(decoder.lm_head.weight.grad)


if __name__ == '__main__':
    unittest.main()
