import subprocess
import pandas as pd
import os
import glob
import time

# 1. Backbones to benchmark
MODELS = [
    "resnet50",
    "vit_base_patch16_224",
    "mobilenet_v3_large",
    "vgg16_bn",
    "convnext_tiny",
    "densenet121",
    "swin_tiny_patch4_window7_224",
    "efficientnet_v2_s"
]

# 2. Configuration
DATASET = "rsna_boneage"
SPLIT = "training"  # Change to "validation" to run the other split
VARIANT = "SLOC_m"
# Set num_images to -1 to run the ENTIRE split
NUM_IMAGES = -1 

def run_benchmarks():
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"STARTING BENCHMARK: {model} on {SPLIT} split")
        print(f"{'='*60}")
        
        # subprocess.run isolates GPU memory for each model run
        cmd = [
            "python", "run.py",
            "--variant", VARIANT,
            "--model", model,
            "--dataset", DATASET,
            "--split", SPLIT,
            "--num_images", str(NUM_IMAGES),
            "--resume"
        ]
        
        start = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = (time.time() - start) / 3600
            print(f"Finished {model} in {elapsed:.2f} hours.")
        except subprocess.CalledProcessError as e:
            print(f"Error encountered during {model}: {e}")

def generate_leaderboard():
    print(f"\n{'='*60}")
    print(f"GENERATING LEADERBOARD FOR {SPLIT.upper()} SPLIT")
    print(f"{'='*60}")
    
    # Matches the output naming convention in your main() block
    pattern = f"sloc_{VARIANT.lower()}_*_{DATASET}_{SPLIT}_results.csv"
    all_files = glob.glob(pattern)
    summary_data = []

    for file in all_files:
        try:
            # Extract model name from filename logic
            # sloc_sloc_m_<model>_rsna_boneage_training_results.csv
            parts = file.replace(".csv", "").split("_")
            # find index between variant and dataset
            model_name = parts[2] 
            
            df = pd.read_csv(file)
            means = df.mean(numeric_only=True).to_dict()
            means['Model'] = model_name
            summary_data.append(means)
        except Exception as e:
            print(f"Could not process file {file}: {e}")

    if summary_data:
        leaderboard = pd.DataFrame(summary_data)
        # Metrics strictly preserved
        cols = ['Model', 'IDD', 'AIC', 'SIC', 'NPD', 'DEL', 'INS', 'POS', 'NEG']
        # Filter columns that exist
        final_cols = [c for c in cols if c in leaderboard.columns]
        leaderboard = leaderboard[final_cols].sort_values(by='IDD', ascending=False)
        
        print("\nRANKED LEADERBOARD:")
        print(leaderboard.to_string(index=False))
        leaderboard.to_csv(f"final_leaderboard_{SPLIT}.csv", index=False)
    else:
        print("No result files found for aggregation.")

if __name__ == "__main__":
    run_benchmarks()
    generate_leaderboard()