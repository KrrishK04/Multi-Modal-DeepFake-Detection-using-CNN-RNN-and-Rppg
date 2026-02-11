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

from dfd import run_deepfake_detection # Reuse the exact pipeline/fusion logic


def list_video_files(directory_path: str) -> List[str]:
    """Scans a directory and returns a list of video file paths."""
    try:
        return [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    except Exception:
        return []


def sample_videos(files: List[str], sample_size: int, seed: int = 42) -> List[str]:
    """Selects a random, reproducible sample of files from a list."""
    rng = random.Random(seed)
    if len(files) <= sample_size:
        rng.shuffle(files)
        return files
    return rng.sample(files, sample_size)


def evaluate_on_celebdf(
    celebd_df_root: str = r"C:\Users\KK\Desktop\dfd\Celeb-DF",
    real_subdir: str = "Celeb-real",
    fake_subdir: str = "Celeb-synthesis",
    num_real: int = 100,
    num_fake: int = 100,
    model_path: str = "model_89_acc_40_frames_final_data.pt",
    run_rppg: bool = True,
    show_visualization: bool = False,
    sequence_length: int = 40,
    frame_rate: int = 40,
    batch_size: int = 30,
    signal_size: int = 270,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Evaluates the deepfake detection model on a sample of the Celeb-DF dataset,
    calculates metrics, and generates an ROC curve.
    """
    real_dir = os.path.join(celebd_df_root, real_subdir)
    fake_dir = os.path.join(celebd_df_root, fake_subdir)

    real_files = list_video_files(real_dir)
    fake_files = list_video_files(fake_dir)

    sampled_real = sample_videos(real_files, num_real, seed)
    sampled_fake = sample_videos(fake_files, num_fake, seed + 1)

    results_rows: List[Dict[str, str]] = []
    counters = {
        "real_total": len(sampled_real),
        "fake_total": len(sampled_fake),
        "real_correct": 0,
        "fake_correct": 0,
    }
    
    # Lists to store data for the ROC curve
    y_true: List[int] = []
    y_scores: List[float] = []

    start_time = time.time()

    def process_file(file_path: str, label: str) -> None:
        """Helper function to process a single video file."""
        try:
            result = run_deepfake_detection(
                video_path=file_path,
                model_path=model_path,
                run_rppg=run_rppg,
                show_visualization=show_visualization,
                sequence_length=sequence_length,
                frame_rate=frame_rate,
                batch_size=batch_size,
                signal_size=signal_size,
            )
        except Exception as e:
            result = None
            print(f"Error processing {file_path}: {e}")

        pred = result["prediction"] if result and "prediction" in result else "Unknown"
        conf = float(result["confidence"]) if result and "confidence" in result else 0.0
        
        # Update counters for accuracy
        is_correct = 1 if (label.upper() == pred.upper()) else 0
        if label.upper() == "REAL":
            counters["real_correct"] += is_correct
            y_true.append(0) # 0 for the 'negative' class
        elif label.upper() == "FAKE":
            counters["fake_correct"] += is_correct
            y_true.append(1) # 1 for the 'positive' class
        
        # Calculate score for the "positive" class (FAKE) for ROC curve
        score = conf if pred.upper() == "FAKE" else 1.0 - conf
        y_scores.append(score)

        # Store detailed results for the CSV file
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

    print(f"Found {len(real_files)} real and {len(fake_files)} fake videos.")
    print(f"Sampling {len(sampled_real)} real and {len(sampled_fake)} fake videos for evaluation.\n")

    # --- Main Processing Loops ---
    for idx, fpath in enumerate(sampled_real, 1):
        print(f"[REAL {idx}/{len(sampled_real)}] {fpath}")
        process_file(fpath, label="REAL")

    for idx, fpath in enumerate(sampled_fake, 1):
        print(f"[FAKE {idx}/{len(sampled_fake)}] {fpath}")
        process_file(fpath, label="FAKE")

    duration_min = (time.time() - start_time) / 60.0

    # --- 1. Aggregate Accuracy Metrics ---
    real_acc = counters["real_correct"] / counters["real_total"] if counters["real_total"] > 0 else 0.0
    fake_acc = counters["fake_correct"] / counters["fake_total"] if counters["fake_total"] > 0 else 0.0
    total_videos = counters["real_total"] + counters["fake_total"]
    total_correct = counters["real_correct"] + counters["fake_correct"]
    overall_acc = total_correct / total_videos if total_videos > 0 else 0.0

    # --- 2. Generate and Save ROC Curve Plot ---
    roc_auc = 0.0
    if y_true and y_scores:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right"); plt.grid(True)
        
        # Save in three different formats
        plt.savefig(os.path.join(CURRENT_DIR, "roc_curve.pdf"), format="pdf", bbox_inches="tight")
        plt.savefig(os.path.join(CURRENT_DIR, "roc_curve_transparent.png"), format="png", transparent=True, bbox_inches="tight")
        plt.savefig(os.path.join(CURRENT_DIR, "roc_curve.png"), format="png", bbox_inches="tight")
        plt.close()

    # --- 3. Save CSV and Print Final Summary ---
    csv_path = os.path.join(CURRENT_DIR, "celebdf_batch_results.csv")
    if results_rows:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
            writer.writeheader()
            writer.writerows(results_rows)

    print("\nEvaluation complete.")
    print(f"Results saved to: {csv_path}")
    if roc_auc > 0: print("ROC curve plots saved successfully.")

    print(
        f"Real accuracy: {real_acc*100:.2f}% ({counters['real_correct']}/{counters['real_total']}), "
        f"Fake accuracy: {fake_acc*100:.2f}% ({counters['fake_correct']}/{counters['fake_total']}), "
        f"Overall: {overall_acc*100:.2f}%, AUC: {roc_auc:.2f}, Time: {duration_min:.1f} min"
    )

    return {
        "real_accuracy": real_acc,
        "fake_accuracy": fake_acc,
        "overall_accuracy": overall_acc,
        "minutes": duration_min,
        "csv": csv_path,
        "auc": roc_auc,
    }

# --- Main execution block ---
if __name__ == '__main__':
    res = evaluate_on_celebdf(
        celebd_df_root=r"C:\Users\KK\Desktop\dfd\Celeb-DF",
        real_subdir="Celeb-real",
        fake_subdir="Celeb-synthesis",
        num_real=100,
        num_fake=100,
        model_path=os.path.join(CURRENT_DIR, "checkpoint.pt"),
        run_rppg=True,
        show_visualization=False,
        sequence_length=20,
        frame_rate=30,
        batch_size=30,
        signal_size=270,
        seed=42,
    )