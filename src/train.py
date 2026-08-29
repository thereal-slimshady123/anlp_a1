"""
Training and Evaluation Pipeline for Sequence-to-Sequence Transformer Ablation Study (C1–C5).
Integrates Pre-LN Transformer, WandB tracking, greedy decoding validation,
Hugging Face Hub checkpointing, and performance metrics.
"""

import os
import sys
import time
import math
import json
import argparse
import random
from typing import Optional, Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt

# Set UTF-8 encoding and real-time line buffering for Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
except Exception:
    pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.transformer import Seq2SeqTransformer
from src.dataset import (
    get_dataloaders,
    PAD_ID,
    SOS_ID,
    EOS_ID,
    ByteTokenizer
)
from src.utils import evaluate_dataset

# Enable TensorFloat-32 (TF32) matmuls for faster execution on Ampere/Lovelace GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Linear warmup over warmup_steps followed by smooth cosine annealing decay to 10% of peak learning rate.
    Starts immediately at 10% of peak LR (e.g. 0.0001 - 0.0005) and ramps to full peak LR within warmup_steps.
    """
    def lr_lambda(step: int):
        if step < warmup_steps:
            return max(0.1, float(step + 1) / float(max(1, warmup_steps)))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.05, 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))
        
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def get_memory_usage(device: torch.device) -> float:
    """Returns peak memory allocated in MB (CUDA or CPU)."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        # Fallback for CPU
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0


def train_one_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_wandb: bool = False,
    wandb_run: Any = None,
    log_interval: int = 20
) -> Dict[str, float]:
    """
    Trains model for one full epoch with optional Automatic Mixed Precision (AMP).
    """
    model.train()
    total_loss = 0.0
    total_tokens = 0
    step_times = []
    use_amp = (device.type == 'cuda' and scaler is not None)
    
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        t0 = time.time()
        
        src = batch["src"].to(device, non_blocking=True)
        tgt_in = batch["tgt_in"].to(device, non_blocking=True)
        tgt_out = batch["tgt_out"].to(device, non_blocking=True)
        src_mask = batch["src_mask"].to(device, non_blocking=True)
        tgt_mask = batch["tgt_mask"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with torch.amp.autocast('cuda'):
                logits = model(src=src, tgt=tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
                vocab_size = logits.size(-1)
                loss = criterion(logits.view(-1, vocab_size), tgt_out.contiguous().view(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(src=src, tgt=tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
            vocab_size = logits.size(-1)
            loss = criterion(logits.view(-1, vocab_size), tgt_out.contiguous().view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        if scheduler is not None:
            scheduler.step()
            
        t1 = time.time()
        step_duration = (t1 - t0) * 1000.0  # ms
        step_times.append(step_duration)
        
        # Count non-pad tokens for accurate token loss
        non_pad_tokens = (tgt_out != PAD_ID).sum().item()
        total_loss += loss.item() * non_pad_tokens
        total_tokens += non_pad_tokens
        
        if (step + 1) % log_interval == 0 or (step + 1) == len(train_loader):
            current_lr = optimizer.param_groups[0]['lr']
            peak_mem = get_memory_usage(device)
            avg_step_t = np.mean(step_times[-log_interval:])
            
            print(f"Epoch [{epoch+1}] | Step [{step+1}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | LR: {current_lr:.6f} | "
                  f"Step Time: {avg_step_t:.1f}ms | Peak Mem: {peak_mem:.1f}MB")
            
            if use_wandb and wandb_run is not None:
                wandb_run.log({
                    "train/step_loss": loss.item(),
                    "train/lr": current_lr,
                    "train/step_time_ms": avg_step_t,
                    "train/peak_memory_mb": peak_mem,
                    "train/step": epoch * len(train_loader) + step
                })
                
    avg_epoch_loss = total_loss / max(1, total_tokens)
    epoch_duration = time.time() - start_time
    
    return {
        "train_loss": avg_epoch_loss,
        "train_perplexity": math.exp(min(avg_epoch_loss, 20.0)),
        "epoch_time_sec": epoch_duration,
        "avg_step_time_ms": np.mean(step_times) if step_times else 0.0,
        "peak_memory_mb": get_memory_usage(device)
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config_name: str,
    metrics: Dict[str, Any],
    output_dir: str = "outputs",
    hf_repo_id: Optional[str] = None,
    upload_hf: bool = False
):
    """
    Saves model checkpoint locally and optionally pushes to Hugging Face Hub.
    """
    ckpt_dir = os.path.join(output_dir, "checkpoints", config_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    best_path = os.path.join(ckpt_dir, "model_best.pt")
    
    checkpoint = {
        "epoch": epoch + 1,
        "config_name": config_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    torch.save(checkpoint, best_path)
    print(f"[CHECKPOINT] Saved best checkpoint: {best_path}")
    
    if upload_hf and hf_repo_id:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=hf_repo_id, repo_type="model", exist_ok=True)
            api.upload_file(
                path_or_fileobj=best_path,
                path_in_repo=f"{config_name}/model_best.pt",
                repo_id=hf_repo_id,
                repo_type="model"
            )
            print(f"[HF HUB] Uploaded checkpoint to Hugging Face Hub: https://huggingface.co/{hf_repo_id}")
        except Exception as e:
            print(f"[WARNING] Failed to upload checkpoint to HF Hub: {e}")


def plot_training_results(
    history: Dict[str, List[float]],
    config_name: str,
    output_dir: str
):
    """
    Plots and saves loss and evaluation metrics curves into outputs/.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Ablation Experiment: {config_name} Training Curves", fontsize=16)
    
    # 1. Loss & Perplexity
    axes[0, 0].plot(epochs, history["train_loss"], 'b-o', label="Train Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross Entropy Loss")
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # 2. BLEU & ROUGE
    axes[0, 1].plot(epochs, history["bleu"], 'g-s', label="BLEU-4")
    axes[0, 1].plot(epochs, history["rougeL"], 'm-^', label="ROUGE-L")
    axes[0, 1].set_title("BLEU & ROUGE-L Scores (%)")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score (%)")
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # 3. Accuracies
    axes[1, 0].plot(epochs, history["bit_accuracy"], 'c-d', label="Bit-Level Acc")
    axes[1, 0].plot(epochs, history["sequence_accuracy"], 'r-x', label="Seq Exact Match")
    axes[1, 0].set_title("Accuracies (%)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy (%)")
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    # 4. Levenshtein Distance & Similarity
    axes[1, 1].plot(epochs, history["levenshtein_similarity"], 'orange', marker='o', label="Levenshtein Sim (%)")
    axes[1, 1].set_title("Levenshtein Edit Similarity")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Similarity (%)")
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"{config_name}_training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved training curves plot: {plot_path}")


def run_training(args):
    """
    Main training routine for a given configuration.
    """
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    # Auto-tune cuDNN for fastest algorithms on this GPU
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    print(f"\n=======================================================")
    print(f"[EXPERIMENT] Launching Ablation Configuration: {args.config.upper()}")
    print(f"[DEVICE] {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) | Seed: {args.seed}")
    print(f"=======================================================\n")
    
    # 1. Initialize WandB if enabled
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"run-{args.config.upper()}",
                config=vars(args)
            )
            print(f"[WANDB] Initialized: {args.wandb_project}/run-{args.config.upper()}")
        except Exception as e:
            print(f"[WARNING] WandB initialization failed: {e}. Continuing without WandB.")
            args.wandb = False
            
    # 2. Prepare DataLoaders
    print("[DATA] Loading datasets and tokenizers...")
    train_loader, val_loader, src_tok, tgt_tok = get_dataloaders(
        cipher_file=args.cipher_file,
        plain_file=args.plain_file,
        config_name=args.config,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=args.output_dir,
        cipher_vocab_size=args.cipher_vocab_size,
        plain_vocab_size=args.plain_vocab_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda')
    )
    
    if args.config.upper() == 'C5':
        src_vocab_size = src_tok.get_vocab_size()
        tgt_vocab_size = tgt_tok.get_vocab_size()
    else:
        src_vocab_size = src_tok.get_vocab_size()
        tgt_vocab_size = tgt_tok.get_vocab_size()
        
    print(f"[VOCAB] Source Vocab: {src_vocab_size} | Target Vocab: {tgt_vocab_size}")
    print(f"[BATCHES] Train: {len(train_loader)} | Val: {len(val_loader)}")
    
    # 3. Instantiate Model
    model = Seq2SeqTransformer.from_config_name(
        config_name=args.config,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        d_ff=args.d_ff,
        num_kv_heads=args.num_kv_heads,
        dropout=args.dropout,
        blt_patch_size=args.blt_patch_size,
        blt_d_byte=args.blt_d_byte
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total Parameters: {total_params:,}")
    
    if args.wandb and wandb_run is not None:
        wandb_run.summary["total_parameters"] = total_params
        
    # 4. Optimizer, Scheduler, Loss
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay)
    
    total_training_steps = len(train_loader) * args.epochs
    scheduler = get_lr_scheduler(optimizer, warmup_steps=args.warmup_steps, total_steps=total_training_steps)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 5. Training and Evaluation Loop
    history = {
        "train_loss": [],
        "val_loss": [],
        "epoch_time": [],
        "bit_accuracy": [],
        "sequence_accuracy": [],
        "levenshtein_distance": [],
        "levenshtein_similarity": [],
        "bleu": [],
        "rouge1": [],
        "rouge2": [],
        "rougeL": []
    }
    
    best_bleu = -1.0
    best_val_metrics = {}
    best_epoch = 0
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        print(f"\n=======================================================")
        print(f" Epoch [{epoch+1}/{args.epochs}] Starting")
        print(f"=======================================================")
        
        # Train 1 epoch
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            scaler=scaler,
            use_wandb=args.wandb,
            wandb_run=wandb_run,
            log_interval=args.log_interval
        )
        
        # Evaluate on validation set
        print("[VAL] Running validation & greedy decoding...")
        val_metrics = evaluate_dataset(
            model=model,
            dataloader=val_loader,
            tgt_tokenizer=tgt_tok,
            device=device,
            criterion=criterion,
            max_decode_len=args.max_decode_len,
            max_eval_batches=args.max_eval_batches
        )
        
        total_epoch_time = time.time() - epoch_start_time
        
        print(f"\n-------------------------------------------------------")
        print(f" [EPOCH {epoch+1}/{args.epochs} COMPLETED] Time: {total_epoch_time:.2f}s")
        print(f"   Train Loss:      {train_metrics['train_loss']:.4f} (Perplexity: {train_metrics['train_perplexity']:.2f})")
        print(f"   Val Loss:        {val_metrics['val_loss']:.4f}")
        print(f"   Bit-Level Acc:   {val_metrics['bit_accuracy']:.2f}%")
        print(f"   Seq Match Acc:   {val_metrics['sequence_accuracy']:.2f}%")
        print(f"   Levenshtein Sim: {val_metrics['levenshtein_similarity']:.2f}% (Dist: {val_metrics['levenshtein_distance']:.1f})")
        print(f"   BLEU-4:          {val_metrics['bleu']:.2f}%")
        print(f"   ROUGE-L:         {val_metrics['rougeL']:.2f}%")
        print(f"   Sample Pred:     {val_metrics['sample_pred'][:100]}...")
        print(f"   Sample Target:   {val_metrics['sample_tgt'][:100]}...")
        print(f"-------------------------------------------------------\n")
        
        # Update history
        history["train_loss"].append(train_metrics["train_loss"])
        history["val_loss"].append(val_metrics["val_loss"])
        history["epoch_time"].append(total_epoch_time)
        history["bit_accuracy"].append(val_metrics["bit_accuracy"])
        history["sequence_accuracy"].append(val_metrics["sequence_accuracy"])
        history["levenshtein_distance"].append(val_metrics["levenshtein_distance"])
        history["levenshtein_similarity"].append(val_metrics["levenshtein_similarity"])
        history["bleu"].append(val_metrics["bleu"])
        history["rouge1"].append(val_metrics["rouge1"])
        history["rouge2"].append(val_metrics["rouge2"])
        history["rougeL"].append(val_metrics["rougeL"])
        
        # Log epoch metrics to WandB
        if args.wandb and wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train/loss_epoch": train_metrics["train_loss"],
                "val/loss": val_metrics["val_loss"],
                "val/epoch_time_sec": total_epoch_time,
                "val/bit_accuracy": val_metrics["bit_accuracy"],
                "val/sequence_accuracy": val_metrics["sequence_accuracy"],
                "val/levenshtein_distance": val_metrics["levenshtein_distance"],
                "val/levenshtein_similarity": val_metrics["levenshtein_similarity"],
                "val/bleu": val_metrics["bleu"],
                "val/rouge1": val_metrics["rouge1"],
                "val/rouge2": val_metrics["rouge2"],
                "val/rougeL": val_metrics["rougeL"]
            })
            
        # Checkpoint locally on best BLEU
        if val_metrics["bleu"] > best_bleu:
            best_bleu = val_metrics["bleu"]
            best_val_metrics = val_metrics
            best_epoch = epoch
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config_name=args.config.upper(),
                metrics=val_metrics,
                output_dir=args.output_dir,
                hf_repo_id=args.hf_repo,
                upload_hf=False
            )
            
    # Final Hugging Face Hub upload of the best model
    if args.hf_repo:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=best_epoch,
            config_name=args.config.upper(),
            metrics=best_val_metrics if best_val_metrics else val_metrics,
            output_dir=args.output_dir,
            hf_repo_id=args.hf_repo,
            upload_hf=True
        )
        
    # Save training curves plot and metrics JSON
    plot_training_results(history, args.config.upper(), args.output_dir)
    
    metrics_json_path = os.path.join(args.output_dir, f"{args.config.upper()}_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": args.config.upper(),
            "best_bleu": best_bleu,
            "final_metrics": history
        }, f, indent=2)
    print(f"[METRICS] Saved metrics summary: {metrics_json_path}")
    
    if args.wandb and wandb_run is not None:
        wandb_run.finish()
        
    print(f"\n Finished training for {args.config.upper()}! Best BLEU: {best_bleu:.2f}%\n")


def main():
    parser = argparse.ArgumentParser(description="Train Transformer Ablation Configurations (C1–C5)")
    
    # Configuration & Datasets
    parser.add_argument("--config", type=str, default="C1", choices=["C1", "C2", "C3", "C4", "C5"],
                        help="Ablation configuration to train (C1-C5)")
    parser.add_argument("--cipher_file", type=str, default="dataset/brown_cipher.txt",
                        help="Path to encrypted binary ciphertext dataset")
    parser.add_argument("--plain_file", type=str, default="dataset/brown_plain.txt",
                        help="Path to plaintext target English dataset")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Directory to save checkpoints, plots, and tokenizer models")
    
    # Model Architecture Hyperparameters
    parser.add_argument("--d_model", type=int, default=256, help="Transformer hidden dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=4, help="Number of decoder layers")
    parser.add_argument("--d_ff", type=int, default=512, help="FeedForward inner dimension")
    parser.add_argument("--num_kv_heads", type=int, default=None, help="Number of KV heads for GQA (C3)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # BLT Specific Hyperparameters (C5)
    parser.add_argument("--blt_patch_size", type=int, default=4, help="Patch size for BLT local encoder/decoder")
    parser.add_argument("--blt_d_byte", type=int, default=64, help="Byte embedding dimension for BLT")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Linear warmup steps")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation data split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Vocab sizes
    parser.add_argument("--cipher_vocab_size", type=int, default=1000, help="BPE vocab size for ciphertext")
    parser.add_argument("--plain_vocab_size", type=int, default=4000, help="BPE vocab size for English plaintext")
    
    # Evaluation & Hardware
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader worker processes (0 recommended on Windows with in-memory datasets)")
    parser.add_argument("--max_decode_len", type=int, default=128, help="Maximum length for greedy decoding")
    parser.add_argument("--max_eval_batches", type=int, default=1, help="Limit number of validation batches for greedy decoding speed")
    parser.add_argument("--max_samples", type=int, default=None, help="Debug mode: limit total dataset samples")
    parser.add_argument("--log_interval", type=int, default=20, help="Steps between training logs")
    parser.add_argument("--no_cuda", action="store_true", help="Force CPU training")
    
    # Tracking & Checkpointing (Enabled by default)
    parser.add_argument("--no_wandb", action="store_true", help="Disable Weights & Biases tracking")
    parser.add_argument("--wandb_project", type=str, default="anlp-assignment1", help="WandB project name")
    parser.add_argument("--hf_repo", type=str, default="goofymonsieur123/anlp-assignment1", help="Hugging Face Hub repository ID to upload weights")
    parser.add_argument("--no_hf", action="store_true", help="Disable Hugging Face Hub checkpoint upload")
    
    args = parser.parse_args()
    
    # Process boolean flags
    args.wandb = not args.no_wandb
    if args.no_hf or args.hf_repo in ("None", "none", ""):
        args.hf_repo = None
        
    # Auto-set 4 KV heads for GQA (C3) if not specified
    if args.config.upper() == 'C3' and args.num_kv_heads is None:
        args.num_kv_heads = 4
        
    # Auto-adjust batch size for C5 BLT to optimize memory if using default
    if args.config.upper() == 'C5' and args.batch_size == 32:
        args.batch_size = 16
        
    run_training(args)


if __name__ == "__main__":
    main()
