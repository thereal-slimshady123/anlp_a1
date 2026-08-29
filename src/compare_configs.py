import os
import json
import glob
from typing import Dict, Any

def main():
    # Look for metrics relative to this script's directory in the outputs folder
    outputs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs"))
    
    # Find all JSON metrics files
    metrics_files = glob.glob(os.path.join(outputs_dir, "*_metrics.json"))
    
    if not metrics_files:
        print(f"No metrics files found in {outputs_dir}.")
        print("Please run train.py first to generate metrics.")
        return
        
    print(f"Found {len(metrics_files)} configuration metrics files.")
    
    configs_data = []
    
    for file_path in sorted(metrics_files):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            config_name = data.get("config", os.path.basename(file_path).replace("_metrics.json", ""))
            best_bleu = data.get("best_bleu", 0.0)
            history = data.get("final_metrics", {})
            
            # Extract final epoch values (last item in history lists)
            def get_final(key: str, default=0.0):
                val_list = history.get(key, [])
                return val_list[-1] if val_list else default
                
            train_loss = get_final("train_loss")
            val_loss = get_final("val_loss")
            bit_acc = get_final("bit_accuracy")
            seq_acc = get_final("sequence_accuracy")
            lev_sim = get_final("levenshtein_similarity")
            
            # Calculate total time if epoch_time is recorded
            epoch_times = history.get("epoch_time", [])
            total_time_sec = sum(epoch_times) if epoch_times else 0.0
            
            # Convert time to MM:SS or HH:MM:SS
            if total_time_sec > 3600:
                hours = int(total_time_sec // 3600)
                minutes = int((total_time_sec % 3600) // 60)
                seconds = int(total_time_sec % 60)
                time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                minutes = int(total_time_sec // 60)
                seconds = int(total_time_sec % 60)
                time_str = f"{minutes:02d}:{seconds:02d}"
                
            configs_data.append({
                "config": config_name,
                "best_bleu": best_bleu,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bit_acc": bit_acc,
                "seq_acc": seq_acc,
                "lev_sim": lev_sim,
                "time": time_str
            })
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}")
            
    if not configs_data:
        print("No valid metrics data could be parsed.")
        return
        
    # Print Markdown Table
    print("\n### Configuration Performance Comparison\n")
    headers = [
        "Config", "Best BLEU-4 (%)", "Final Train Loss", "Final Val Loss",
        "Bit Accuracy (%)", "Seq Match Acc (%)", "Levenshtein Sim (%)", "Train Time"
    ]
    
    # Calculate column widths
    row_format = "| {:<8} | {:<16} | {:<16} | {:<14} | {:<16} | {:<18} | {:<19} | {:<10} |"
    divider = "|-" + "-|-".join(["-" * 8, "-" * 16, "-" * 16, "-" * 14, "-" * 16, "-" * 18, "-" * 19, "-" * 10]) + "-|"
    
    print(row_format.format(*headers))
    print(divider)
    
    for row in configs_data:
        print(row_format.format(
            row["config"],
            f"{row['best_bleu']:.2f}%",
            f"{row['train_loss']:.4f}",
            f"{row['val_loss']:.4f}",
            f"{row['bit_acc']:.2f}%",
            f"{row['seq_acc']:.2f}%",
            f"{row['lev_sim']:.2f}%",
            row["time"]
        ))
    print()

if __name__ == "__main__":
    main()
