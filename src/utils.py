"""
Evaluation Metrics and Utility Functions.
Implements:
1. Autoregressive Greedy Decoding.
2. Bit-Level Accuracy.
3. Sequence Accuracy (Exact Match).
4. Levenshtein Distance & Normalized Edit Similarity.
5. Corpus & Sentence BLEU Score.
6. ROUGE-1, ROUGE-2, and ROUGE-L Scores.
7. Validation Evaluation Loop.
"""

import math
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
import Levenshtein
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from src.dataset import PAD_ID, SOS_ID, EOS_ID, UNK_ID, make_causal_mask, ByteTokenizer


def compute_bit_accuracy(preds: List[str], targets: List[str]) -> float:
    """
    Computes bit-level accuracy by converting predicted and target strings to binary (8 bits per char)
    and calculating the fraction of matching bits.
    """
    if not preds or not targets:
        return 0.0
        
    total_acc = 0.0
    for pred, target in zip(preds, targets):
        try:
            pred_bytes = pred.encode('utf-8', errors='ignore')
            target_bytes = target.encode('utf-8', errors='ignore')
        except Exception:
            pred_bytes = pred.encode('ascii', errors='ignore')
            target_bytes = target.encode('ascii', errors='ignore')
            
        pred_bits = "".join(f"{b:08b}" for b in pred_bytes)
        target_bits = "".join(f"{b:08b}" for b in target_bytes)
        
        max_len = max(len(pred_bits), len(target_bits))
        if max_len == 0:
            total_acc += 1.0
            continue
            
        min_len = min(len(pred_bits), len(target_bits))
        matching_bits = sum(1 for p, t in zip(pred_bits[:min_len], target_bits[:min_len]) if p == t)
        total_acc += matching_bits / max_len
        
    return total_acc / len(preds)


def compute_sequence_accuracy(preds: List[str], targets: List[str]) -> float:
    """
    Computes Exact Match Sequence Accuracy.
    Returns fraction of samples where prediction exactly matches target (whitespace-invariant).
    """
    if not preds or not targets:
        return 0.0
        
    exact_matches = 0
    for p, t in zip(preds, targets):
        p_words = p.strip().split()
        t_words = t.strip().split()
        if p_words and t_words and p_words == t_words:
            exact_matches += 1
            
    return exact_matches / len(preds)


def compute_levenshtein(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    """
    Computes raw Levenshtein edit distance and normalized similarity score [0, 1].
    
    Returns:
        Tuple[float, float]: (mean_raw_distance, mean_normalized_similarity)
    """
    if not preds or not targets:
        return 0.0, 0.0
        
    raw_distances = []
    normalized_similarities = []
    
    for p, t in zip(preds, targets):
        dist = Levenshtein.distance(p, t)
        raw_distances.append(dist)
        
        max_l = max(len(p), len(t), 1)
        sim = 1.0 - (dist / max_l)
        normalized_similarities.append(max(0.0, sim))
        
    return (
        sum(raw_distances) / len(raw_distances),
        sum(normalized_similarities) / len(normalized_similarities)
    )


def compute_bleu_and_rouge(preds: List[str], targets: List[str]) -> Dict[str, float]:
    """
    Computes BLEU-4 (with smoothing) and ROUGE-1, ROUGE-2, ROUGE-L F1 scores.
    
    Returns:
        Dict[str, float]: {'bleu': ..., 'rouge1': ..., 'rouge2': ..., 'rougeL': ...}
    """
    if not preds or not targets:
        return {'bleu': 0.0, 'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        
    # 1. BLEU Score
    smooth = SmoothingFunction().method1
    bleu_scores = []
    for p, t in zip(preds, targets):
        ref_tokens = [t.split()] if t.strip() else [[]]
        hyp_tokens = p.split() if p.strip() else []
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth)
        bleu_scores.append(score)
        
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    
    # 2. ROUGE Scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_list, rouge2_list, rougeL_list = [], [], []
    
    for p, t in zip(preds, targets):
        res = scorer.score(target=t, prediction=p)
        rouge1_list.append(res['rouge1'].fmeasure)
        rouge2_list.append(res['rouge2'].fmeasure)
        rougeL_list.append(res['rougeL'].fmeasure)
        
    return {
        'bleu': avg_bleu * 100.0,  # Scaled to 0-100%
        'rouge1': (sum(rouge1_list) / len(rouge1_list)) * 100.0,
        'rouge2': (sum(rouge2_list) / len(rouge2_list)) * 100.0,
        'rougeL': (sum(rougeL_list) / len(rougeL_list)) * 100.0
    }


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int = 256,
    repetition_penalty: float = 1.15,
    sos_id: int = SOS_ID,
    eos_id: int = EOS_ID,
    pad_id: int = PAD_ID
) -> torch.Tensor:
    """
    Performs autoregressive greedy decoding with optional repetition penalty for a batch of source sequences.
    
    Args:
        model: Seq2SeqTransformer instance.
        src: (batch_size, src_seq_len)
        src_mask: (batch_size, src_seq_len)
        max_len: Maximum target generation length. Default: 256.
        repetition_penalty: Penalty factor applied to previously generated tokens. Default: 1.15.
        
    Returns:
        torch.Tensor: Generated token IDs of shape (batch_size, generated_len)
    """
    model.eval()
    device = src.device
    batch_size = src.size(0)
    
    # 1. Encode source sequence once
    memory, eff_src_mask = model.encode(src, src_mask=src_mask)
    
    mem_mask_4d = None
    if eff_src_mask is not None:
        if eff_src_mask.dim() == 2:
            mem_mask_4d = eff_src_mask.unsqueeze(1).unsqueeze(2)
        else:
            mem_mask_4d = eff_src_mask
            
    # 2. Initialize target tensor with <sos>
    ys = torch.full((batch_size, 1), fill_value=sos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    for _ in range(max_len):
        cur_len = ys.size(1)
        tgt_mask = make_causal_mask(cur_len).to(device)
        
        # Decode current prefix
        dec_out = model.decode(
            tgt=ys,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=mem_mask_4d
        )
        
        if model.is_blt:
            logits = model.blt_decoder(dec_out, target_byte_len=cur_len)
        else:
            logits = model.output_projection(dec_out)
            
        # Get next token logits for the last position: (batch_size, vocab_size)
        next_token_logits = logits[:, -1, :].clone()
        
        # Apply repetition penalty to previously generated tokens
        if repetition_penalty != 1.0:
            mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
            mask.scatter_(1, ys, True)
            mask[:, pad_id] = False
            mask[:, sos_id] = False
            mask[:, eos_id] = False
            
            next_token_logits = torch.where(
                mask,
                torch.where(next_token_logits > 0, next_token_logits / repetition_penalty, next_token_logits * repetition_penalty),
                next_token_logits
            )
                            
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
        
        # If sequence already finished, emit pad_id
        next_token = torch.where(finished.unsqueeze(1), torch.full_like(next_token, pad_id), next_token)
        ys = torch.cat([ys, next_token], dim=1)
        
        # Update finished mask
        finished = finished | (next_token.squeeze(1) == eos_id)
        if finished.all():
            break
            
    return ys


def decode_tokens_to_text(token_ids: List[List[int]], tokenizer: Any) -> List[str]:
    """
    Decodes generated token ID lists into readable text strings using either BPE or ByteTokenizer.
    """
    texts = []
    is_byte = isinstance(tokenizer, ByteTokenizer)
    
    for ids in token_ids:
        # Strip <sos>, <eos>, <pad>
        cleaned = []
        for tid in ids:
            if tid == SOS_ID or tid == PAD_ID:
                continue
            if tid == EOS_ID:
                break
            cleaned.append(tid)
            
        if is_byte:
            text = tokenizer.decode(cleaned)
        else:
            text = tokenizer.decode(cleaned)
        texts.append(text)
        
    return texts


def evaluate_dataset(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tgt_tokenizer: Any,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_decode_len: int = 128,
    max_eval_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Runs full evaluation over a validation dataloader using greedy decoding.
    
    Computes:
    - val_loss (CrossEntropy validation loss)
    - bit_acc (Bit/character accuracy)
    - seq_acc (Exact match sequence accuracy)
    - lev_dist (Mean Levenshtein distance)
    - lev_sim (Mean normalized Levenshtein similarity %)
    - bleu (BLEU-4 score %)
    - rouge1, rouge2, rougeL (ROUGE F1 scores %)
    """
    model.eval()
    all_preds: List[str] = []
    all_targets: List[str] = []
    val_loss = 0.0
    val_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_eval_batches is not None and i >= max_eval_batches:
                break
                
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            raw_targets = batch["raw_tgt"]
            
            # Compute validation cross-entropy loss
            if criterion is not None:
                logits = model(src=src, tgt=tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
                vocab_size = logits.size(-1)
                loss = criterion(logits.view(-1, vocab_size), tgt_out.contiguous().view(-1))
                non_pad_tokens = (tgt_out != PAD_ID).sum().item()
                val_loss += loss.item() * non_pad_tokens
                val_tokens += non_pad_tokens
            
            gen_ids = greedy_decode(
                model=model,
                src=src,
                src_mask=src_mask,
                max_len=max_decode_len
            )
            
            gen_texts = decode_tokens_to_text(gen_ids.cpu().tolist(), tgt_tokenizer)
            all_preds.extend(gen_texts)
            all_targets.extend(raw_targets)
            
    avg_val_loss = val_loss / max(1, val_tokens) if val_tokens > 0 else 0.0
            
    # Compute all metrics
    bit_acc = compute_bit_accuracy(all_preds, all_targets) * 100.0
    seq_acc = compute_sequence_accuracy(all_preds, all_targets) * 100.0
    lev_dist, lev_sim = compute_levenshtein(all_preds, all_targets)
    bleu_rouge = compute_bleu_and_rouge(all_preds, all_targets)
    
    return {
        "val_loss": avg_val_loss,
        "bit_accuracy": bit_acc,
        "sequence_accuracy": seq_acc,
        "levenshtein_distance": lev_dist,
        "levenshtein_similarity": lev_sim * 100.0,
        "bleu": bleu_rouge["bleu"],
        "rouge1": bleu_rouge["rouge1"],
        "rouge2": bleu_rouge["rouge2"],
        "rougeL": bleu_rouge["rougeL"],
        "sample_pred": all_preds[0] if all_preds else "",
        "sample_tgt": all_targets[0] if all_targets else ""
    }
