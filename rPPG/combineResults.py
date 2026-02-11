import csv
import os
from collections import defaultdict

# --- CONFIGURATION ---
# The script will look for these files in the same directory.
FILE_1 = 'ffpp_batch_results.csv'
FILE_2 = 'ffpp_batch_results1.csv'
FILE_3 = 'ffpp_batch_results2.csv'
FILE_4 = 'ffpp_batch_results3.csv'
# --- END CONFIGURATION ---

def load_unique_videos(filenames: list[str]) -> dict:
    """
    Reads one or more CSV files and returns a dictionary of unique videos.
    The file path is used as the key to ensure deduplication.
    """
    processed_videos = {}  # Use a dict {file_path: row_data}
    
    for filename in filenames:
        try:
            with open(filename, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                file_count = 0
                for row in reader:
                    file_path = row.get('file')
                    if not file_path:
                        print(f"Warning: Skipping row with no file path in {filename}")
                        continue
                    
                    # Use file path as unique key.
                    # If a video is in both files, the last one read wins.
                    processed_videos[file_path] = row
                    file_count += 1
            print(f"Successfully loaded {file_count} rows from {filename}.")
        
        except FileNotFoundError:
            print(f"Warning: File not found: {filename}. Skipping.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return processed_videos

def analyze_results(processed_videos: dict):
    """
    Analyzes the combined video data and prints a detailed accuracy report.
    """
    
    # Use defaultdict to automatically initialize counters
    # stats will look like: {'FaceSwap': {'total': 0, 'correct': 0}, ...}
    stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    
    for row in processed_videos.values():
        try:
            file_path = row['file']
            # The 'correct' column is a string '1' or '0'
            is_correct = int(row['correct'])
            
            # Extract the category (e.g., 'FaceSwap', 'original')
            # os.path.dirname(file_path) -> .../FaceForensics++_C23/FaceSwap
            # os.path.basename(...) -> FaceSwap
            category = os.path.basename(os.path.dirname(file_path))
            
            # Update stats for that category
            stats[category]['total'] += 1
            stats[category]['correct'] += is_correct
            
        except Exception as e:
            print(f"Warning: Skipping bad row {row}: {e}")
    
    # --- Print Report ---
    print("\n" + "="*40)
    print("--- ACCURACY REPORT PER CATEGORY ---")
    print("="*44)
    
    overall_fake_total = 0
    overall_fake_correct = 0
    overall_real_total = 0
    overall_real_correct = 0

    # Sort stats for clean printing (alphabetical by category)
    for category in sorted(stats.keys()):
        data = stats[category]
        total = data['total']
        correct = data['correct']
        
        if total == 0: continue
        
        accuracy = (correct / total) * 100
        # Print with aligned columns
        print(f"Category: {category:<18} | Accuracy: {accuracy:>6.2f}% ({correct:>4}/{total:<4})")
        
        # Aggregate totals based on the folder name
        if category.lower() == 'original' or category.lower() == 'real':
            overall_real_total += total
            overall_real_correct += correct
        else:
            # Assume any other category is a fake type
            overall_fake_total += total
            overall_fake_correct += correct
    
    # --- Print Summary ---
    print("\n" + "="*40)
    print("--- OVERALL SUMMARY ---")
    print("="*44)
    
    if overall_real_total > 0:
        real_acc = (overall_real_correct / overall_real_total) * 100
        print(f"ALL REAL Videos:           | Accuracy: {real_acc:>6.2f}% ({overall_real_correct:>4}/{overall_real_total:<4})")
    
    if overall_fake_total > 0:
        fake_acc = (overall_fake_correct / overall_fake_total) * 100
        print(f"ALL FAKE Videos (Combined):| Accuracy: {fake_acc:>6.2f}% ({overall_fake_correct:>4}/{overall_fake_total:<4})")

    total_all = overall_real_total + overall_fake_total
    correct_all = overall_real_correct + overall_fake_correct
    
    if total_all > 0:
        all_acc = (correct_all / total_all) * 100
        print("-" * 44)
        print(f"FINAL TOTAL:               | Accuracy: {all_acc:>6.2f}% ({correct_all:>4}/{total_all:<4})")

def main():
    """Main function to run the analysis."""
    filenames = [FILE_1, FILE_2, FILE_3, FILE_4]
    unique_videos = load_unique_videos(filenames)
    
    if not unique_videos:
        print("No video data found. Check CSV filenames. Exiting.")
        return
        
    print(f"\nAnalyzing {len(unique_videos)} unique videos from all files...")
    analyze_results(unique_videos)

if __name__ == "__main__":
    main()
