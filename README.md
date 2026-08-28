# Custom Sequence-to-Sequence Transformer Framework & Ablation Study

An implementation of a custom Sequence-to-Sequence Transformer architecture built entirely from scratch in PyTorch (without high-level modules like `nn.Transformer`, `nn.MultiheadAttention`, or `nn.LayerNorm`), designed to translate encrypted binary ciphertext strings into plaintext English.

---

## 📁 Repository Structure

```text
2024111009_assignment1/
├── src/
│   ├── models/
│   │   ├── __init__.py        # Exports all low-level building blocks and model classes
│   │   ├── attention.py       # Scaled Dot-Product, MultiHeadAttention (MHA), GroupedQueryAttention (GQA)
│   │   ├── positional.py      # Sinusoidal Absolute Positional Encoding & Rotary Position Embedding (RoPE)
│   │   ├── norm.py            # Custom LayerNorm & RMSNorm
│   │   ├── blt.py             # Byte Latent Transformer (BLT) local encoder/decoder patch modules
│   │   └── transformer.py     # Pre-LN Transformer blocks, Encoder, Decoder, & unified Seq2SeqTransformer
│   ├── dataset.py             # Learned BPE tokenizers, byte-level dataset, dynamic padding & masking
│   ├── train.py               # Pre-LN training loop, WandB tracking, HF checkpointing, plotting
│   └── utils.py               # Metrics: Bit-Level Acc, Seq Exact Match, Levenshtein, BLEU, ROUGE
├── tests/
│   └── test_modules.py        # Comprehensive unit tests for all custom modules and configurations
├── dataset/
│   ├── brown_cipher.txt       # Encrypted binary strings (5,000 parallel lines)
│   ├── brown_plain.txt        # Plaintext English sentences (5,000 parallel lines)
│   └── README.md
├── outputs/                   # Tokenizers, saved checkpoints, and evaluation curve plots
├── requirements.txt
└── prompt.md
```

---

## 🔬 Ablation Configurations (C1–C5)

| Configuration | Positional Encoding | Attention Mechanism | Normalization | Tokenization / Architecture |
| :--- | :--- | :--- | :--- | :--- |
| **C1 (Base)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | LayerNorm | Learned Subword BPE |
| **C2** | Rotary Position Embedding (RoPE) | Multi-Head Attention (MHA) | LayerNorm | Learned Subword BPE |
| **C3** | Sinusoidal Absolute | Grouped-Query Attention (GQA) | LayerNorm | Learned Subword BPE |
| **C4** | Sinusoidal Absolute | Multi-Head Attention (MHA) | RMSNorm | Learned Subword BPE |
| **C5 (BLT)** | Sinusoidal Absolute | Multi-Head Attention (MHA) | LayerNorm | **Token-Free BLT Local Patching** |

---

## 🚀 Installation & Setup

1. **Activate Virtual Environment:**
   ```bash
   # Windows PowerShell
   .\myenv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Login to WandB and Hugging Face (One-time setup):**
   ```bash
   wandb login
   hf auth login
   ```

---

## 🧪 Running Unit Tests

Run the complete unit test suite verifying tensor shapes, RoPE relative shift invariance, GQA key-value broadcasting, causal masking, BLT local patch pooling/unpooling, and end-to-end forward/backward passes for C1–C5:

```bash
python -m unittest tests/test_modules.py
```

---

## 🏋️ Running Experiments (Automatic WandB & Hugging Face Checkpointing)

> **All defaults are configured:**
> - **WandB Tracking:** Automatically active (project: `anlp-assignment1`)
> - **Hugging Face Hub:** Automatically uploads checkpoints to `goofymonsieur123/anlp-assignment1`
> - **Hyperparameters:** `epochs=10`, `d_model=256`, `num_heads=8`, `d_ff=1024`, `lr=5e-4`

### Run Individual Configurations:

```bash
# Configuration 1 (C1 - Baseline: Sinusoidal PE + MHA + LayerNorm + Subword BPE)
python src/train.py --config C1

# Configuration 2 (C2 - RoPE: Rotary Position Embedding + MHA + LayerNorm + Subword BPE)
python src/train.py --config C2

# Configuration 3 (C3 - GQA: Sinusoidal PE + Grouped-Query Attention + LayerNorm + Subword BPE)
python src/train.py --config C3

# Configuration 4 (C4 - RMSNorm: Sinusoidal PE + MHA + RMSNorm + Subword BPE)
python src/train.py --config C4

# Configuration 5 (C5 - BLT: Token-Free Byte Latent Transformer with Local Patching)
python src/train.py --config C5
```

---

## ⚡ Run All 5 Ablations Sequentially (PowerShell)

To run the entire ablation study from start to finish with one command:

```powershell
python src/train.py --config C1; python src/train.py --config C2; python src/train.py --config C3; python src/train.py --config C4; python src/train.py --config C5
```

---

## 🛠️ Optional Override Flags

If you ever want to customize or disable tracking:
- `--no_wandb`: Turn off Weights & Biases logging.
- `--no_hf`: Turn off Hugging Face Hub checkpoint uploading.
- `--epochs <N>`: Change number of epochs (default: 10).
- `--batch_size <N>`: Change batch size (default: 32 for C1-C4, 16 for C5).
- `--lr <LR>`: Change learning rate (default: 5e-4).

---

## 📊 Evaluation Metrics & Logged Outputs

### Logged Metrics:
- **Training Loss & Perplexity** (step-level and epoch-level)
- **Peak Memory Usage (MB)** & **Average Step Time (ms)**
- **Bit-Level Accuracy (%)**: Token/character-level exact match rate.
- **Sequence Accuracy (%)**: 100% sentence exact reconstruction rate.
- **Levenshtein Distance & Normalized Similarity (%)**: Minimum character edit distance.
- **BLEU-4 Score (%)**: Overlap precision with smoothing.
- **ROUGE-1, ROUGE-2, ROUGE-L (%)**: Recall & F1 overlaps.

### Saved Artifacts:
- Checkpoints saved in `outputs/checkpoints/{CONFIG}/model_best.pt`.
- Training curves saved as PNG plots in `outputs/{CONFIG}_training_curves.png`.
- Tokenizer models saved in `outputs/tokenizer_cipher.json` and `outputs/tokenizer_plain.json`.
