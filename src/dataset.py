"""
Dataset and Tokenization Pipeline for Encrypted Binary-to-Plaintext Translation.
100% From-Scratch Implementation of:
1. BPETokenizer: Pure Python Byte-Pair Encoding (BPE) subword tokenizer (no external tokenizer libraries).
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

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID = 0
SOS_ID = 1
EOS_ID = 2
UNK_ID = 3



class BPETokenizer:
    """
    Pure Python Subword Tokenizer implementing the Byte-Pair Encoding (BPE) algorithm from scratch.
    (Sennrich et al., 2016).
    
    Zero third-party library dependencies (no Hugging Face `tokenizers` / `sentencepiece`).
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
        
        # Merge rules: Dict[(str, str), int] mapping symbol pair to its merge priority rank
        self.merges: Dict[Tuple[str, str], int] = {}
        
        # Cache for fast word encoding
        self._cache: Dict[str, List[str]] = {}

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def _get_stats(self, vocab_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Counts frequency of all adjacent symbol pairs across the vocabulary."""
        pairs = collections.defaultdict(int)
        for word_tuple, freq in vocab_freqs.items():
            for i in range(len(word_tuple) - 1):
                pairs[(word_tuple[i], word_tuple[i + 1])] += freq
        return pairs

    def _merge_vocab(
        self,
        pair: Tuple[str, str],
        vocab_freqs: Dict[Tuple[str, ...], int]
    ) -> Dict[Tuple[str, ...], int]:
        """Replaces all occurrences of pair (sym1, sym2) with concatenated symbol in vocabulary."""
        new_vocab = {}
        first, second = pair
        merged_sym = first + second

        for word_tuple, freq in vocab_freqs.items():
            new_word = []
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_word.append(merged_sym)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(
        self,
        corpus: List[str],
        vocab_size: int = 1000,
        min_frequency: int = 2,
        is_binary: bool = False
    ):
        """
        Trains BPE subword merges iteratively from a list of text strings.
        
        Args:
            corpus: List of text sentences / binary strings.
            vocab_size: Target vocabulary size (including special tokens).
            min_frequency: Minimum occurrence count for a pair to be merged.
            is_binary: If True, chunks binary sequences for fast n-gram bit merge discovery.
        """
        print(f"[BPE] Training tokenizer from scratch (Target Vocab: {vocab_size}, is_binary={is_binary})...")
        
        # 1. Count word frequencies in corpus
        word_counts = collections.defaultdict(int)
        base_chars: Set[str] = set()

        sample_corpus = corpus[:1000] if (is_binary and len(corpus) > 1000) else corpus

        for line in sample_corpus:
            line = line.strip()
            if not line:
                continue
            if is_binary:
                # Chunk binary stream into 32-char windows so BPE discovers variable-length bit subwords
                for j in range(0, len(line), 32):
                    chunk = line[j:j+32]
                    word_counts[chunk] += 1
                    base_chars.update(chunk)
            else:
                # Natural English: split by whitespace and preserve words
                words = re.findall(r"\w+|[^\w\s]", line, re.UNICODE)
                for w in words:
                    word_counts[w] += 1
                    base_chars.update(w)

        # 2. Add all unique base characters to vocabulary
        for char in sorted(base_chars):
            if char not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[char] = new_id
                self.id_to_token[new_id] = char

        # 3. Represent each word as a tuple of base characters
        word_freqs: Dict[Tuple[str, ...], int] = {
            tuple(char for char in word): freq
            for word, freq in word_counts.items()
        }

        # 4. Initialize global pair counts and pair-to-words index mapping
        pair_counts = collections.defaultdict(int)
        pair_to_words = collections.defaultdict(set)
        
        for symbols, freq in word_freqs.items():
            for pair in zip(symbols[:-1], symbols[1:]):
                pair_counts[pair] += freq
                pair_to_words[pair].add(symbols)

        # 5. Iteratively learn BPE merge operations
        num_merges = vocab_size - len(self.vocab)
        for i in range(num_merges):
            if not pair_counts:
                break
                
            best_pair = max(pair_counts, key=pair_counts.get)
            best_freq = pair_counts[best_pair]
            
            if best_freq < min_frequency:
                break
                
            # Record merge rank and add merged token to vocabulary
            self.merges[best_pair] = i
            merged_token = best_pair[0] + best_pair[1]
            
            if merged_token not in self.vocab:
                new_id = len(self.vocab)
                self.vocab[merged_token] = new_id
                self.id_to_token[new_id] = merged_token
                
            # Retrieve words containing the best_pair
            words_to_update = list(pair_to_words[best_pair])
            
            # Update only the affected words
            for symbols in words_to_update:
                freq = word_freqs[symbols]
                
                # Decrement pair counts of the old representation
                for pair in zip(symbols[:-1], symbols[1:]):
                    pair_counts[pair] -= freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_words[pair].discard(symbols)
                    if not pair_to_words[pair]:
                        del pair_to_words[pair]
                    
                # Perform the merge
                new_symbols = []
                j = 0
                while j < len(symbols):
                    if j < len(symbols) - 1 and symbols[j] == best_pair[0] and symbols[j+1] == best_pair[1]:
                        new_symbols.append(merged_token)
                        j += 2
                    else:
                        new_symbols.append(symbols[j])
                        j += 1
                new_symbols = tuple(new_symbols)
                
                # Update word frequency representation
                del word_freqs[symbols]
                word_freqs[new_symbols] = word_freqs.get(new_symbols, 0) + freq
                
                # Increment pair counts of the new representation
                for pair in zip(new_symbols[:-1], new_symbols[1:]):
                    pair_counts[pair] += freq
                    pair_to_words[pair].add(new_symbols)

        # Pre-sort merges by rank for fast encoding
        self._sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        print(f"[BPE] Training complete! Learned {len(self.merges)} merges. Total Vocab Size: {len(self.vocab)}")

    def _bpe_word(self, word: str) -> List[str]:
        """Applies learned BPE merge rules to a single word/chunk using pre-sorted merges."""
        if word in self._cache:
            return self._cache[word]

        word_tuple = list(word)
        if len(word_tuple) <= 1:
            return word_tuple

        # Apply merges in priority order (lowest rank first) — single pass per merge
        for (first, second), _ in self._sorted_merges:
            if first not in word_tuple:
                continue
            i = 0
            new_word = []
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and word_tuple[i] == first and word_tuple[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word_tuple[i])
                    i += 1
            word_tuple = new_word
            if len(word_tuple) == 1:
                break

        self._cache[word] = word_tuple
        return word_tuple

    def encode(self, text: str, is_binary: bool = False) -> Any:
        """
        Encodes a string into a list of token IDs using learned BPE subwords.
        Returns an object with `.ids` and `.tokens` attributes for interface compatibility.
        """
        text = text.strip()
        tokens = []
        
        if is_binary:
            # Chunk binary stream into 32-char windows and apply BPE
            for j in range(0, len(text), 32):
                chunk = text[j:j+32]
                tokens.extend(self._bpe_word(chunk))
        else:
            # Natural English words + punctuation
            words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
            for w in words:
                tokens.extend(self._bpe_word(w))

        # Convert token strings to integer IDs
        ids = [self.vocab.get(t, self.unk_id) for t in tokens]

        class Encoding:
            def __init__(self, ids, tokens):
                self.ids = ids
                self.tokens = tokens

        return Encoding(ids, tokens)

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a readable string."""
        tokens = []
        for tid in ids:
            if tid in (self.pad_id, self.sos_id, self.eos_id):
                continue
            token_str = self.id_to_token.get(tid, self.unk_token)
            tokens.append(token_str)
            
        # For natural English, join words with space where appropriate
        result = ""
        for t in tokens:
            if re.match(r"^[01]+$", t):
                result += t
            elif re.match(r"^\w+$", t):
                result += (" " if result and not result.endswith(" ") else "") + t
            else:
                result += t
        return result.strip()

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
            
        if not isinstance(data, dict) or "vocab" not in data or "merges" not in data:
            raise ValueError(f"Invalid or legacy tokenizer format in {file_path}")
            
        tokenizer.vocab = data["vocab"]
        tokenizer.id_to_token = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.merges = {(item[0], item[1]): item[2] for item in data["merges"]}
        # Pre-sort merges by rank for fast encoding
        tokenizer._sorted_merges = sorted(tokenizer.merges.items(), key=lambda x: x[1])
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
        raw_bytes = text.strip().encode('utf-8')
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


def get_or_train_tokenizers(
    cipher_file: str,
    plain_file: str,
    output_dir: str = "outputs",
    cipher_vocab_size: int = 1000,
    plain_vocab_size: int = 4000
) -> Tuple[BPETokenizer, BPETokenizer]:
    """
    Loads saved from-scratch BPE tokenizers or trains new ones directly on dataset files.
    """
    os.makedirs(output_dir, exist_ok=True)
    cipher_tok_path = os.path.join(output_dir, "tokenizer_cipher.json")
    plain_tok_path = os.path.join(output_dir, "tokenizer_plain.json")
    
    try:
        cipher_tokenizer = BPETokenizer.from_file(cipher_tok_path)
    except Exception:
        with open(cipher_file, 'r', encoding='utf-8') as f:
            cipher_lines = [line.strip() for line in f if line.strip()]
        cipher_tokenizer = BPETokenizer()
        cipher_tokenizer.train(cipher_lines, vocab_size=cipher_vocab_size, is_binary=True)
        cipher_tokenizer.save(cipher_tok_path)
        
    try:
        plain_tokenizer = BPETokenizer.from_file(plain_tok_path)
    except Exception:
        with open(plain_file, 'r', encoding='utf-8') as f:
            plain_lines = [line.strip() for line in f if line.strip()]
        plain_tokenizer = BPETokenizer()
        plain_tokenizer.train(plain_lines, vocab_size=plain_vocab_size, is_binary=False)
        plain_tokenizer.save(plain_tok_path)
        
    return cipher_tokenizer, plain_tokenizer


class CipherPlainDataset(Dataset):
    """
    Subword Tokenized Dataset for C1–C4 configurations.
    Pre-tokenizes into memory for zero DataLoader overhead during training.
    """
    def __init__(
        self,
        cipher_lines: List[str],
        plain_lines: List[str],
        cipher_tokenizer: BPETokenizer,
        plain_tokenizer: BPETokenizer,
        max_src_len: int = 128,
        max_tgt_len: int = 128
    ):
        assert len(cipher_lines) == len(plain_lines), "Line count mismatch between cipher and plain files"
        print(f"[DATA] Pre-tokenizing {len(cipher_lines)} samples into RAM tensors...")
        self.samples = []
        
        for c_line, p_line in zip(cipher_lines, plain_lines):
            c_text = c_line.strip()
            p_text = p_line.strip()
            
            # Tokenize with from-scratch BPE
            src_ids = cipher_tokenizer.encode(c_text, is_binary=True).ids
            tgt_ids = plain_tokenizer.encode(p_text, is_binary=False).ids
            
            # Add <sos> and <eos> tokens
            src_ids = [SOS_ID] + src_ids[:max_src_len - 2] + [EOS_ID]
            tgt_ids = [SOS_ID] + tgt_ids[:max_tgt_len - 2] + [EOS_ID]
            
            self.samples.append({
                "src": torch.tensor(src_ids, dtype=torch.long),
                "tgt": torch.tensor(tgt_ids, dtype=torch.long),
                "raw_src": c_text,
                "raw_tgt": p_text
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


class BLTByteDataset(Dataset):
    """
    Raw Byte-level Dataset for C5 (BLT) configuration.
    Pre-tokenizes into memory for zero DataLoader overhead.
    """
    def __init__(
        self,
        cipher_lines: List[str],
        plain_lines: List[str],
        byte_tokenizer: ByteTokenizer,
        max_src_len: int = 2048,
        max_tgt_len: int = 1024
    ):
        assert len(cipher_lines) == len(plain_lines), "Line count mismatch between cipher and plain files"
        self.samples = []
        
        for c_line, p_line in zip(cipher_lines, plain_lines):
            c_text = c_line.strip()
            p_text = p_line.strip()
            
            src_ids = byte_tokenizer.encode(c_text, is_binary=True).ids
            tgt_ids = byte_tokenizer.encode(p_text, is_binary=False).ids
            
            src_ids = [SOS_ID] + src_ids[:max_src_len - 2] + [EOS_ID]
            tgt_ids = [SOS_ID] + tgt_ids[:max_tgt_len - 2] + [EOS_ID]
            
            self.samples.append({
                "src": torch.tensor(src_ids, dtype=torch.long),
                "tgt": torch.tensor(tgt_ids, dtype=torch.long),
                "raw_src": c_text,
                "raw_tgt": p_text
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
    cipher_vocab_size: int = 1000,
    plain_vocab_size: int = 4000,
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
    
    train_cipher = [cipher_lines[i] for i in train_indices]
    train_plain = [plain_lines[i] for i in train_indices]
    val_cipher = [cipher_lines[i] for i in val_indices]
    val_plain = [plain_lines[i] for i in val_indices]
    
    is_blt = (config_name.upper() == 'C5')
    
    if is_blt:
        src_tok = ByteTokenizer()
        tgt_tok = ByteTokenizer()
        train_dataset = BLTByteDataset(train_cipher, train_plain, src_tok)
        val_dataset = BLTByteDataset(val_cipher, val_plain, src_tok)
    else:
        src_tok, tgt_tok = get_or_train_tokenizers(
            cipher_file=cipher_file,
            plain_file=plain_file,
            output_dir=output_dir,
            cipher_vocab_size=cipher_vocab_size,
            plain_vocab_size=plain_vocab_size
        )
        train_dataset = CipherPlainDataset(train_cipher, train_plain, src_tok, tgt_tok)
        val_dataset = CipherPlainDataset(val_cipher, val_plain, src_tok, tgt_tok)
        
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
