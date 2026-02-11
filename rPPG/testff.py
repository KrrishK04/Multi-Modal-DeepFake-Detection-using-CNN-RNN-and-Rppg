import os
import sys
import csv
import time
import random
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Ensure imports work when running from project root
CURRENT_DIR = os.getcwd()
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# --- ASSUMPTION ---
# This script assumes you have a file named 'dfd.py' in the same directory
# or in the Python path, and it has a function 'run_deepfake_detection'.
try:
    from dfd import run_deepfake_detection
except ImportError:
    print("Error: Could not import 'run_deepfake_detection' from 'dfd'.")
    print("Please ensure 'dfd.py' is in the same directory or in your sys.path.")
    # Define a placeholder function to allow the script to be analyzed
    def run_deepfake_detection(**kwargs):
        print(f"--- FAKE DETECTION RUN (missing dfd.py) ---")
        time.sleep(0.1)
        # Return a plausible-looking dummy result
        is_fake = random.choice([True, False])
        conf = random.random()
        return {
            "prediction": "FAKE" if is_fake else "REAL",
            "confidence": conf,
            "cnn_rnn_prediction": "FAKE" if is_fake else "REAL",
            "cnn_rnn_confidence": conf,
            "rppg_prediction": "FAKE" if random.choice([True, False]) else "REAL",
            "rppg_confidence": random.random(),
            "heart_rate": random.uniform(60, 90) if not is_fake else 0.0,
            "method": "Fusion"
        }

def list_video_files(directory_path: str) -> List[str]:
    """Scans a directory and returns a list of video file paths."""
    try:
        return [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    except Exception as e:
        print(f"Error listing files in {directory_path}: {e}")
        return []


def sample_videos(files: List[str], sample_size: int, seed: int = 42) -> List[str]:
    """Selects a random, reproducible sample of files from a list."""
    rng = random.Random(seed)
    if not files:
        print("Warning: No files provided to sample_videos.")
        return []
    if len(files) <= sample_size:
        print(f"Warning: Requested sample size ({sample_size}) is >= total files ({len(files)}). Returning all files shuffled.")
        rng.shuffle(files)
        return files
    return rng.sample(files, sample_size)


def _plot_roc_curve(y_true: List[int], y_scores: List[float], output_filename_base: str, title: str) -> float:
    """Helper function to generate and save an ROC curve plot."""
    roc_auc = 0.0
    if not y_true or not y_scores:
        print(f"Warning: No data for ROC curve '{title}'. Skipping plot.")
        return roc_auc
        
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True)
        
        plt.savefig(os.path.join(CURRENT_DIR, f"{output_filename_base}.pdf"), format="pdf", bbox_inches="tight")
        plt.savefig(os.path.join(CURRENT_DIR, f"{output_filename_base}_transparent.png"), format="png", transparent=True, bbox_inches="tight")
        plt.savefig(os.path.join(CURRENT_DIR, f"{output_filename_base}.png"), format="png", bbox_inches="tight")
        plt.close()
        print(f"ROC curve plots saved to '{output_filename_base}.[pdf/png]'")
        
    except Exception as e:
        print(f"Error generating ROC plot: {e}")
        
    return roc_auc


def _process_video_file(
    file_path: str,
    label: str,
    counters: Dict,
    y_true: List[int],
    y_scores: List[float],
    results_rows: List[Dict],
    model_params: Dict
) -> None:
    """Internal helper to process one video and update all tracking lists."""
    try:
        result = run_deepfake_detection(
            video_path=file_path,
            model_path=model_params["model_path"],
            run_rppg=model_params["run_rppg"],
            show_visualization=model_params["show_visualization"],
            sequence_length=model_params["sequence_length"],
            frame_rate=model_params["frame_rate"],
            batch_size=model_params["batch_size"],
            signal_size=model_params["signal_size"],
        )
    except Exception as e:
        print(f"\n--- !!! ---")
        print(f"CRITICAL ERROR processing {file_path}: {e}")
        print(f"This file will be SKIPPED. Continuing with next file.")
        print(f"--- !!! ---\n")
        result = None
        
        # We need to adjust the counters so we don't penalize accuracy
        if label.upper() == "REAL":
            counters["real_total"] -= 1
        elif label.upper() == "FAKE":
            counters["fake_total"] -= 1
        return # Skip the rest of this function for this file

    pred = result["prediction"] if result and "prediction" in result else "Unknown"
    conf = float(result["confidence"]) if result and "confidence" in result else 0.0
    
    is_correct = 1 if (label.upper() == pred.upper()) else 0
    if label.upper() == "REAL":
        counters["real_correct"] += is_correct
        y_true.append(0) # 0 for 'real'
    elif label.upper() == "FAKE":
        counters["fake_correct"] += is_correct
        y_true.append(1) # 1 for 'fake'
    
    score = conf if pred.upper() == "FAKE" else 1.0 - conf
    y_scores.append(score)

    results_rows.append(
        {
            "file": file_path, "label": label, "final_prediction": pred,
            "final_confidence": f"{conf:.2f}",
            "cnn_pred": result.get("cnn_rnn_prediction", "Unknown") if result else "Unknown",
            "cnn_conf": f"{float(result.get('cnn_rnn_confidence', 0.0)):.2f}" if result else "0.00",
            "rppg_pred": result.get("rppg_prediction", "Unknown") if result else "Unknown",
            "rppg_conf": f"{float(result.get('rppg_confidence', 0.0)):.2f}" if result else "0.00",
            "heart_rate": f"{float(result.get('heart_rate', 0.0)):.2f}" if result else "0.00",
            "fusion_method": result.get("method", "") if result else "",
            "correct": str(is_correct),
        }
    )


def _save_csv_results(csv_path: str, results_rows: List[Dict]) -> None:
    """Helper function to save the detailed results to a CSV file."""
    if not results_rows:
        print("No results to save to CSV.")
        return
        
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"Results saved to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV to {csv_path}: {e}")


def _print_summary(counters: Dict, roc_auc: float, duration_min: float) -> Dict:
    """Helper function to print the final summary and return the metrics dict."""
    real_acc = counters["real_correct"] / counters["real_total"] if counters["real_total"] > 0 else 0.0
    fake_acc = counters["fake_correct"] / counters["fake_total"] if counters["fake_total"] > 0 else 0.0
    total_videos = counters["real_total"] + counters["fake_total"]
    total_correct = counters["real_correct"] + counters["fake_correct"]
    overall_acc = total_correct / total_videos if total_videos > 0 else 0.0

    print(
        f"\nReal accuracy: {real_acc*100:.2f}% ({counters['real_correct']}/{counters['real_total']}), "
        f"Fake accuracy: {fake_acc*100:.2f}% ({counters['fake_correct']}/{counters['fake_total']}), "
        f"Overall: {overall_acc*100:.2f}%, AUC: {roc_auc:.2f}, Time: {duration_min:.1f} min"
    )

    return {
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "overall_accuracy": overall_acc,
        "minutes": duration_min,
        "auc": roc_auc,
    }


def evaluate_on_ffpp(
    ffpp_root: str,
    real_subdir: str,
    num_real: int,
    fake_sampling_config: Dict[str, int], # MODIFIED: Takes a dict
    model_path: str,
    run_rppg: bool,
    show_visualization: bool,
    sequence_length: int,
    frame_rate: int,
    batch_size: int,
    signal_size: int,
    seed: int,
    save_interval: int = 50, # <<<--- NEW PARAMETER
) -> Dict[str, float]:
    """
    Evaluates the model on FaceForensics++ with per-folder sampling.
    """
    print("\n--- Starting Evaluation on FaceForensics++ ---")
    
    # --- 1. Process REAL Videos ---
    real_dir = os.path.join(ffpp_root, real_subdir)
    real_files = list_video_files(real_dir)
    sampled_real = sample_videos(real_files, num_real, seed)
    print(f"Found {len(real_files)} real videos, sampling {len(sampled_real)}.")

    # --- 2. Process FAKE Videos (Based on config dict) ---
    sampled_fake = []
    rng_seed = seed + 1 # Use a rolling seed for reproducibility
    
    if not fake_sampling_config:
        print("Warning: 'fake_sampling_config' is empty. No fake videos will be processed.")
    
    print("\n--- Sampling Fake Videos ---")
    for subdir, num_to_sample in fake_sampling_config.items():
        if num_to_sample == 0:
            print(f"Skipping {subdir} (0 videos requested).")
            continue

        fake_dir = os.path.join(ffpp_root, subdir)
        all_files_in_subdir = list_video_files(fake_dir)
        
        if not all_files_in_subdir:
            print(f"Warning: No videos found in {fake_dir} for category '{subdir}'")
            continue

        print(f"Sampling {num_to_sample} videos from {subdir} (found {len(all_files_in_subdir)} total)...")
        
        # Use `sample_videos` helper, with a unique seed for this subdir
        sampled_for_this_subdir = sample_videos(
            all_files_in_subdir, 
            num_to_sample, 
            seed=rng_seed
        )
        
        sampled_fake.extend(sampled_for_this_subdir)
        rng_seed += 1 # Increment seed for the next folder
    
    print("------------------------------\n")
    
    # Shuffle the final fake list so they are processed in a random order
    final_fake_rng = random.Random(seed)
    final_fake_rng.shuffle(sampled_fake)

    # --- 3. Set up Counters and Run Evaluation ---
    results_rows: List[Dict[str, str]] = []
    counters = {
        "real_total": len(sampled_real), "fake_total": len(sampled_fake),
        "real_correct": 0, "fake_correct": 0,
    }
    y_true: List[int] = []
    y_scores: List[float] = []
    
    # <<<--- CSV PATH MOVED HERE ---
    csv_path = os.path.join(CURRENT_DIR, "ffpp_batch_results.csv")
    
    start_time = time.time()
    videos_processed_count = 0 # <<<--- NEW COUNTER

    model_params = {
        "model_path": model_path, "run_rppg": run_rppg,
        "show_visualization": show_visualization, "sequence_length": sequence_length,
        "frame_rate": frame_rate, "batch_size": batch_size, "signal_size": signal_size,
    }

    print(f"Total videos to process: {len(sampled_real)} REAL, {len(sampled_fake)} FAKE.\n")
    print(f"Auto-saving results every {save_interval} videos to: {csv_path}\n")

    # --- Main Processing Loops ---
    for idx, fpath in enumerate(sampled_real, 1):
        print(f"[REAL {idx}/{len(sampled_real)}] {fpath}")
        _process_video_file(fpath, "REAL", counters, y_true, y_scores, results_rows, model_params)
        
        # <<<--- NEW AUTO-SAVE LOGIC ---
        videos_processed_count += 1
        if videos_processed_count > 0 and videos_processed_count % save_interval == 0:
            print(f"\n--- Auto-saving results after {videos_processed_count} videos... ---")
            _save_csv_results(csv_path, results_rows)
            print("--- Save complete. Resuming... ---\n")
        # <<<-------------------------

    for idx, fpath in enumerate(sampled_fake, 1):
        print(f"[FAKE {idx}/{len(sampled_fake)}] {fpath}")
        _process_video_file(fpath, "FAKE", counters, y_true, y_scores, results_rows, model_params)
        
        # <<<--- NEW AUTO-SAVE LOGIC ---
        videos_processed_count += 1
        if videos_processed_count > 0 and videos_processed_count % save_interval == 0:
            print(f"\n--- Auto-saving results after {videos_processed_count} videos... ---")
            _save_csv_results(csv_path, results_rows)
            print("--- Save complete. Resuming... ---\n")
        # <<<-------------------------

    duration_min = (time.time() - start_time) / 60.0

    # --- 4. Metrics, Plotting, and Saving ---
    roc_auc = _plot_roc_curve(y_true, y_scores, "ffpp_roc_curve", "ROC Curve - FaceForensics++")
    
    # This final save catches any remaining results
    print("\nSaving final results...")
    _save_csv_results(csv_path, results_rows) 
    
    print("\nFaceForensics++ Evaluation complete.")
    summary = _print_summary(counters, roc_auc, duration_min)
    summary["csv"] = csv_path
    return summary


# --- Main execution block ---
if __name__ == '__main__':
    
    # --- !!! This path is set based on your input !!! ---
    FFPP_DATASET_ROOT = r"C:\Users\KK\Desktop\dfd\FaceForensics++_C23"
    MODEL_CHECKPOINT = os.path.join(CURRENT_DIR, "checkpoint.pt")
    
    # --- !!! YOUR MANUAL CONFIGURATION HERE !!! ---
    #
    # Edit the numbers below to control how many videos are sampled
    # from each specific folder.
    #
    MANUAL_SAMPLING_CONFIG = {
        "real_subdir": "original",
        "num_real": 0,  # Number of videos from the 'original' folder
        
        "fake_counts": {
            "Deepfakes": 100,
            "FaceSwap": 100,
            "Face2Face": 100,
            "FaceShifter": 100,
            "NeuralTextures": 100,
            "DeepFakeDetection": 100,
        }
        # The total fake videos will be the sum of all numbers above
    }
    # --- !!! END OF CONFIGURATION !!! ---

    # --- Shared Model Parameters ---
    model_config = {
        "model_path": MODEL_CHECKPOINT,
        "run_rppg": True,
        "show_visualization": False,
        "sequence_length": 40,
        "frame_rate": 40,
        "batch_size": 16, # Lowered from 30 to prevent memory crashes
        "signal_size": 270,
        "seed": 42,
        "save_interval": 50, # <<<--- SET YOUR SAVE INTERVAL HERE
    }

    # --- Run evaluation on FaceForensics++ ---
    res_ffpp = evaluate_on_ffpp(
        ffpp_root=FFPP_DATASET_ROOT,
        
        # Pass the values from your manual config
        real_subdir=MANUAL_SAMPLING_CONFIG["real_subdir"],
        num_real=MANUAL_SAMPLING_CONFIG["num_real"],
        fake_sampling_config=MANUAL_SAMPLING_CONFIG["fake_counts"],
        
        # Pass the rest of the model config
        **model_config
    )

    print("\n" + "="*80)
    print("FACEFORENSICS++ FINAL SUMMARY:")
    print(res_ffpp)
    print("="*80 + "\n")

    print("FaceForensics++ evaluation finished.")