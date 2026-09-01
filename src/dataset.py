"""
Dataset and Tokenization Pipeline for Encrypted Binary-to-Plaintext Translation.
100% From-Scratch Implementation of:
1. BPETokenizer: Pure Python Byte-Pair Encoding (BPE) subword tokenizer with Sennrich-style </w> word boundaries.
2. ByteTokenizer: Pure Python Token-Free Byte Loader for C5 (BLT).
3. PyTorch Dataset and DataLoader classes with dynamic padding, causal masking, and key padding masks.
"""

import os
import re
import json
import collections
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional, Dict, Any, Set

# Special token constants
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
END_OF_WORD = "</w>"

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def binary_to_byte_string(binary_str: str) -> str:
    """
    Groups every 8 bits of a binary string ('0'/'1') into a single byte character.
    Maps byte values (0-255) to clean Unicode characters starting at 0x4E00
    so they behave as individual character symbols for BPE and byte processing.
    """
    remainder = len(binary_str) % 8
    if remainder != 0:
        binary_str = binary_str + "0" * (8 - remainder)
        
    chars = []
    for i in range(0, len(binary_str), 8):
        byte_val = int(binary_str[i:i+8], 2)
        chars.append(chr(0x4E00 + byte_val))
    return "".join(chars)


def byte_string_to_binary(byte_str: str) -> str:
    """
    Converts grouped byte characters back into standard 8-bit binary strings.
    """
    bits = []
    for c in byte_str:
        byte_val = ord(c) - 0x4E00
        if 0 <= byte_val <= 255:
            bits.append(f"{byte_val:08b}")
    return "".join(bits)


class BPETokenizer:
    """
    Pure Python Subword Tokenizer implementing Byte-Pair Encoding (BPE) from scratch.
    Uses Sennrich-style </w> word boundary tokens for lossless, exact text reconstruction.
    Zero third-party library dependencies.
    """
    def __init__(self, unk_token: str = UNK_TOKEN):
        self.unk_token = unk_token
        self.pad_id = PAD_ID
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        
        # Token to ID mapping
        self.vocab: Dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            SOS_TOKEN: SOS_ID,
            EOS_TOKEN: EOS_ID,
            UNK_TOKEN: UNK_ID
        }
        # ID to Token mapping
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        self.merges: Dict[Tuple[str, str], int] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def train(
        self,
        corpus: List[str],
        vocab_size: int = 2000,
        min_frequency: int = 2,
        is_binary: bool = False
    ):
        """
        Trains BPE subword merges iteratively from a list of text strings.
        """
        print(f"[BPE] Training tokenizer from scratch (Target Vocab: {vocab_size})...")
        
        # 1. Read files and count word frequencies
        word_counts = collections.Counter()
        for line in corpus:
            words = line.strip().split()
            for w in words:
                word_counts[w] += 1
                
        # 2. Initialize vocabulary with unique characters + special tokens
        char_counts = collections.Counter()
        for w in word_counts:
            for char in w:
                char_counts[char] += 1
        char_counts[END_OF_WORD] = sum(word_counts.values())
        
        # Add all unique chars to vocab
        for char, _ in char_counts.most_common():
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.id_to_token[idx] = char
                
        # 3. Represent words as list of character/subword symbols
        word_freqs = {}
        for w, count in word_counts.items():
            symbols = tuple(list(w) + [END_OF_WORD])
            word_freqs[symbols] = count
            
        # 4. Initialize global pair counts and pair-to-words index mapping
        pair_counts = collections.defaultdict(int)
        pair_to_words = collections.defaultdict(set)
        
        for symbols, freq in word_freqs.items():
            for pair in zip(symbols[:-1], symbols[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(symbols)

        num_merges = vocab_size - len(self.vocab)
        for i in range(num_merges):
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_frequency:
                break
                
            self.merges[best_pair] = i
            new_symbol = best_pair[0] + best_pair[1]
            
            # Add to vocab mapping
            idx = len(self.vocab)
            self.vocab[new_symbol] = idx
            self.id_to_token[idx] = new_symbol
            
            # Update affected words
            words_to_update = list(pair_to_words[best_pair])
            for symbols in words_to_update:
                freq = word_freqs[symbols]
                
                for pair in zip(symbols[:-1], symbols[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_words[pair].discard(symbols)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]
                    
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and symbols[j] == best_pair[0] and symbols[j+1] == best_pair[1]:
                        new_symbols.append(new_symbol)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                new_symbols = tuple(new_symbols)
                
                del word_freqs[symbols]
                word_freqs[new_symbols] = word_freqs.get(new_symbols, 0) + freq
                
                for pair in zip(new_symbols[:-1], new_symbols[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(new_symbols)
                    
            if len(self.vocab) >= vocab_size:
                break

        self.bpe_ranks = {pair: idx for pair, idx in self.merges.items()}
        print(f"[BPE] Training complete! Learned {len(self.merges)} merges. Total Vocab Size: {len(self.vocab)}")

    def encode(self, text: str, is_binary: bool = False) -> Any:
        """
        Encodes a string into a list of token IDs using learned BPE subwords.
        """
        if not self.bpe_ranks:
            self.bpe_ranks = {pair: idx for pair, idx in self.merges.items()}
            
        words = text.strip().split()
        encoded_ids = []
        
        for w in words:
            symbols = list(w) + [END_OF_WORD]
            
            while len(symbols) > 1:
                pairs = list(zip(symbols[:-1], symbols[1:]))
                best_pair = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
                
                if best_pair not in self.bpe_ranks:
                    break
                    
                new_symbols = []
                j = 0
                p1, p2 = best_pair
                new_symbol = p1 + p2
                while j < len(symbols):
                    if j < len(symbols) - 1 and symbols[j] == p1 and symbols[j+1] == p2:
                        new_symbols.append(new_symbol)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                symbols = new_symbols
                
            for sym in symbols:
                encoded_ids.append(self.vocab.get(sym, self.unk_id))
                
        class Encoding:
            def __init__(self, ids):
                self.ids = ids

        return Encoding(encoded_ids)

    def decode(self, ids: List[int]) -> str:
        """
        Decodes a list of token IDs back into a readable string.
        """
        tokens = [self.id_to_token.get(idx, self.unk_token) for idx in ids if idx not in (self.pad_id, self.sos_id, self.eos_id)]
        raw_str = "".join(tokens)
        decoded = raw_str.replace(END_OF_WORD, " ")
        return decoded.strip()

    def save(self, file_path: str):
        """Saves vocabulary and merge rules to JSON."""
        data = {
            "vocab": self.vocab,
            "merges": [[p[0], p[1], rank] for p, rank in self.merges.items()]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, file_path: str) -> 'BPETokenizer':
        """Loads vocabulary and merge rules from JSON."""
        tokenizer = cls()
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = {int(v) if str(v).isdigit() else v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = {(item[0], item[1]): item[2] for item in data["merges"]}
        tokenizer.bpe_ranks = {pair: idx for pair, idx in tokenizer.merges.items()}
        return tokenizer


class ByteTokenizer:
    """
    Token-free Byte Tokenizer for C5 (BLT).
    Maps raw ASCII/UTF-8 bytes directly to IDs:
        0: <pad>
        1: <sos>
        2: <eos>
        3: <unk>
        4..259: byte values (0..255 + 4)
    Total vocab size: 260
    """
    def __init__(self):
        self.pad_id = PAD_ID
        self.sos_id = SOS_ID
        self.eos_id = EOS_ID
        self.unk_id = UNK_ID
        self.vocab_size = 260

    def encode(self, text: str, is_binary: bool = False) -> Any:
        if len(text) > 0 and ord(text[0]) >= 0x4E00 and ord(text[0]) < 0x4E00 + 256:
            ids = [ord(c) - 0x4E00 + 4 for c in text]
        else:
            raw_bytes = text.strip().encode('utf-8', errors='replace')
            ids = [b + 4 for b in raw_bytes]
            
        class Encoding:
            def __init__(self, ids):
                self.ids = ids
        return Encoding(ids)

    def decode(self, ids: List[int]) -> str:
        byte_list = []
        for i in ids:
            if i in (self.pad_id, self.sos_id, self.eos_id, self.unk_id):
                continue
            if 4 <= i < 260:
                byte_list.append(i - 4)
        return bytes(byte_list).decode('utf-8', errors='replace')

    def get_vocab_size(self) -> int:
        return self.vocab_size


def segment_aligned_pairs(
    cipher_lines: List[str],
    plain_lines: List[str],
    max_chunk_chars: int = 200
) -> Tuple[List[str], List[str]]:
    """
    Slices paragraph-length parallel lines into strictly 8-byte aligned segments (chunk_chars % 8 == 0).
    Because 1 ASCII plaintext character is strictly 1 byte = 8 cipher bits,
    slicing with start_idx % 8 == 0 ensures every segment begins at phase 0 with respect to the
    repeating 8-byte XOR key ('ANLP2026') with 1-to-1 character-to-bit alignment.
    
    Returns:
        (segmented_cipher_chunks, segmented_plain_chunks)
    """
    all_cipher_chunks = []
    all_plain_chunks = []
    
    chunk_chars = max(8, (max_chunk_chars // 8) * 8)
    
    for c_line, p_line in zip(cipher_lines, plain_lines):
        p_line = p_line.strip()
        c_line = c_line.strip()
        if not p_line or not c_line:
            continue
            
        for i in range(0, len(p_line), chunk_chars):
            p_chunk = p_line[i:i+chunk_chars]
            c_chunk = c_line[i*8:(i+len(p_chunk))*8]
            
            if len(p_chunk) > 0 and len(c_chunk) == len(p_chunk) * 8:
                all_cipher_chunks.append(c_chunk)
                all_plain_chunks.append(p_chunk)
                
    return all_cipher_chunks, all_plain_chunks


def get_or_train_tokenizers(
    plain_file: str,
    output_dir: str = "outputs",
    plain_vocab_size: int = 1000,
    force_retrain: bool = False
) -> Tuple[ByteTokenizer, BPETokenizer]:
    """
    Retrieves ByteTokenizer for source cipher bytes (vocab size 260)
    and trains/loads BPETokenizer for English plaintext.
    """
    os.makedirs(output_dir, exist_ok=True)
    plain_tok_path = os.path.join(output_dir, "tokenizer_plain.json")
    
    cipher_tokenizer = ByteTokenizer()
    plain_tokenizer = None

    if not force_retrain:
        try:
            plain_tokenizer = BPETokenizer.from_file(plain_tok_path)
            if len(plain_tokenizer.vocab) != plain_vocab_size or END_OF_WORD not in plain_tokenizer.vocab:
                plain_tokenizer = None
        except Exception:
            plain_tokenizer = None

    if plain_tokenizer is None:
        with open(plain_file, 'r', encoding='utf-8') as f:
            plain_lines = [line.strip() for line in f if line.strip()]
        plain_tokenizer = BPETokenizer()
        plain_tokenizer.train(plain_lines, vocab_size=plain_vocab_size, is_binary=False)
        plain_tokenizer.save(plain_tok_path)
        
    return cipher_tokenizer, plain_tokenizer


class CipherPlainDataset(Dataset):
    """
    Dataset for C1–C4 configurations.
    Source: 8-bit byte grouped tokens (260 vocab).
    Target: BPE subword tokens from plain_tokenizer.
    """
    def __init__(
        self,
        cipher_lines: List[str],
        plain_lines: List[str],
        cipher_tokenizer: ByteTokenizer,
        plain_tokenizer: BPETokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 256
    ):
        assert len(cipher_lines) == len(plain_lines), "Line count mismatch between cipher and plain files"
        self.samples = []
        
        for c_text, p_text in zip(cipher_lines, plain_lines):
            # 1. Source: 8 bits = 1 byte token (0..255 -> 4..259)
            clean_bits = [c for c in c_text if c in ['0', '1']]
            remainder = len(clean_bits) % 8
            if remainder != 0:
                clean_bits += ['0'] * (8 - remainder)
                
            src_byte_tokens = []
            for i in range(0, len(clean_bits), 8):
                byte_val = int("".join(clean_bits[i:i+8]), 2)
                src_byte_tokens.append(byte_val + 4)
                
            src_ids = [SOS_ID] + src_byte_tokens[:max_src_len - 2] + [EOS_ID]
            
            # 2. Target: BPE subwords
            encoded = plain_tokenizer.encode(p_text, is_binary=False)
            tgt_ids = [SOS_ID] + encoded.ids[:max_tgt_len - 2] + [EOS_ID]
            raw_tgt_str = plain_tokenizer.decode(tgt_ids[1:-1])

            self.samples.append({
                "src": torch.tensor(src_ids, dtype=torch.long),
                "tgt": torch.tensor(tgt_ids, dtype=torch.long),
                "raw_src": c_text,
                "raw_tgt": raw_tgt_str
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class BLTByteDataset(Dataset):
    """
    Raw Byte-level Dataset for C5 (BLT) configuration.
    Source: 8-bit byte grouped tokens (260 vocab).
    Target: UTF-8 raw byte tokens (260 vocab).
    """
    def __init__(
        self,
        cipher_lines: List[str],
        plain_lines: List[str],
        byte_tokenizer: ByteTokenizer,
        max_src_len: int = 256,
        max_tgt_len: int = 256
    ):
        assert len(cipher_lines) == len(plain_lines), "Line count mismatch between cipher and plain files"
        self.samples = []
        
        for c_text, p_text in zip(cipher_lines, plain_lines):
            clean_bits = [c for c in c_text if c in ['0', '1']]
            remainder = len(clean_bits) % 8
            if remainder != 0:
                clean_bits += ['0'] * (8 - remainder)
                
            src_byte_tokens = []
            for i in range(0, len(clean_bits), 8):
                byte_val = int("".join(clean_bits[i:i+8]), 2)
                src_byte_tokens.append(byte_val + 4)
                
            src_ids = [SOS_ID] + src_byte_tokens[:max_src_len - 2] + [EOS_ID]
            
            tgt_byte_tokens = [b + 4 for b in p_text.encode('utf-8', errors='replace')]
            tgt_ids = [SOS_ID] + tgt_byte_tokens[:max_tgt_len - 2] + [EOS_ID]
            raw_tgt_str = byte_tokenizer.decode(tgt_ids[1:-1])

            self.samples.append({
                "src": torch.tensor(src_ids, dtype=torch.long),
                "tgt": torch.tensor(tgt_ids, dtype=torch.long),
                "raw_src": c_text,
                "raw_tgt": raw_tgt_str
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def make_causal_mask(seq_len: int) -> torch.Tensor:
    """Creates a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    return mask.unsqueeze(0).unsqueeze(0)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching variable-length sequences."""
    src_list = [item["src"] for item in batch]
    tgt_list = [item["tgt"] for item in batch]
    raw_src = [item["raw_src"] for item in batch]
    raw_tgt = [item["raw_tgt"] for item in batch]
    
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_list,
        batch_first=True,
        padding_value=PAD_ID
    )
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_list,
        batch_first=True,
        padding_value=PAD_ID
    )
    
    tgt_in = tgt_padded[:, :-1]
    tgt_out = tgt_padded[:, 1:]
    
    src_mask = (src_padded != PAD_ID)
    
    tgt_len = tgt_in.size(1)
    causal_mask = make_causal_mask(tgt_len).to(src_padded.device)
    tgt_pad_mask = (tgt_in != PAD_ID).unsqueeze(1).unsqueeze(2)
    tgt_mask = causal_mask & tgt_pad_mask
    
    return {
        "src": src_padded,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "raw_src": raw_src,
        "raw_tgt": raw_tgt
    }


def get_dataloaders(
    cipher_file: str,
    plain_file: str,
    config_name: str = "C1",
    batch_size: int = 32,
    val_split: float = 0.1,
    seed: int = 42,
    output_dir: str = "outputs",
    cipher_vocab_size: int = 260,
    plain_vocab_size: int = 1000,
    max_samples: Optional[int] = None,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, Any, Any]:
    """Creates Train and Validation DataLoaders for any configuration (C1–C5)."""
    with open(cipher_file, 'r', encoding='utf-8') as f:
        cipher_lines = [line.strip() for line in f if line.strip()]
        
    with open(plain_file, 'r', encoding='utf-8') as f:
        plain_lines = [line.strip() for line in f if line.strip()]
        
    if max_samples is not None:
        cipher_lines = cipher_lines[:max_samples]
        plain_lines = plain_lines[:max_samples]
        
    n_total = len(cipher_lines)
    n_val = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    
    gen = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=gen).tolist()
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_cipher_raw = [cipher_lines[i] for i in train_indices]
    train_plain_raw = [plain_lines[i] for i in train_indices]
    val_cipher_raw = [cipher_lines[i] for i in val_indices]
    val_plain_raw = [plain_lines[i] for i in val_indices]
    
    # Space-delimited paragraph segmentation with character offset alignment
    train_cipher, train_plain = segment_aligned_pairs(train_cipher_raw, train_plain_raw, max_chunk_chars=200)
    val_cipher, val_plain = segment_aligned_pairs(val_cipher_raw, val_plain_raw, max_chunk_chars=200)
    
    print(f"[SEGMENTATION] Expanded dataset via synchronized 1-to-1 slicing:")
    print(f"  Train: {len(train_cipher_raw)} raw paragraphs -> {len(train_cipher)} segments")
    print(f"  Val:   {len(val_cipher_raw)} raw paragraphs -> {len(val_cipher)} segments")
    
    is_blt = (config_name.upper() == 'C5')
    
    if is_blt:
        src_tok = ByteTokenizer()
        tgt_tok = ByteTokenizer()
        train_dataset = BLTByteDataset(train_cipher, train_plain, src_tok, max_src_len=256, max_tgt_len=256)
        val_dataset = BLTByteDataset(val_cipher, val_plain, src_tok, max_src_len=256, max_tgt_len=256)
    else:
        src_tok, tgt_tok = get_or_train_tokenizers(
            plain_file=plain_file,
            output_dir=output_dir,
            plain_vocab_size=plain_vocab_size
        )
        train_dataset = CipherPlainDataset(train_cipher, train_plain, src_tok, tgt_tok, max_src_len=256, max_tgt_len=256)
        val_dataset = CipherPlainDataset(val_cipher, val_plain, src_tok, tgt_tok, max_src_len=256, max_tgt_len=256)
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, src_tok, tgt_tok
