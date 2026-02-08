import subprocess
import pandas as pd
import os
import glob

# 1. Define your 8 backbones
# Mix of CNNs, modern Meta-CNNs, and Transformers
MODELS = [
    "resnet50",                  # Baseline CNN
    "vit_base_patch16_224",      # Baseline Transformer
    "mobilenet_v3_large",        # Lightweight CNN
    "vgg16_bn",                  # Classic CNN
    "convnext_tiny",             # Modern CNN
    "densenet121",               # Medical Imaging Standard
    "swin_tiny_patch4_window7_224", # Hierarchical Transformer
    "efficientnet_v2_s"          # Efficient CNN
]

# 2. Benchmark Configuration
DATASET = "rsna"
NUM_IMAGES = 500  # Adjust based on your time constraints
VARIANT = "SLOC_m"

def run_benchmarks():
    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"LAUNCHING BENCHMARK: {model}")
        print(f"{'='*60}")
        
        # Execute run.py as a subprocess to isolate GPU memory
        cmd = [
            "python", "run.py",
            "--variant", VARIANT,
            "--model", model,
            "--dataset", DATASET,
            "--num_images", str(NUM_IMAGES),
            "--resume"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark for {model}: {e}")

def generate_leaderboard():
    print(f"\n{'='*60}")
    print("GENERATING FINAL LEADERBOARD")
    print(f"{'='*60}")
    
    all_files = glob.glob(f"sloc_{VARIANT.lower()}_*_{DATASET}_results.csv")
    summary_data = []

    for file in all_files:
        try:
            # Extract model name from filename
            model_name = file.split('_')[2:-2]
            model_name = "_".join(model_name)
            
            df = pd.read_csv(file)
            # Calculate means for the metrics
            means = df.mean(numeric_only=True).to_dict()
            means['Model'] = model_name
            summary_data.append(means)
        except Exception as e:
            print(f"Could not process {file}: {e}")

    if summary_data:
        leaderboard = pd.DataFrame(summary_data)
        # Reorder columns to show important metrics first
        cols = ['Model', 'IDD', 'AIC', 'SIC', 'NPD', 'DEL', 'INS', 'POS', 'NEG']
        leaderboard = leaderboard[cols].sort_values(by='IDD', ascending=False)
        
        print("\nFINAL RESULTS (Ranked by IDD):")
        print(leaderboard.to_string(index=False))
        leaderboard.to_csv("final_benchmark_leaderboard.csv", index=False)
    else:
        print("No result files found to aggregate.")

if __name__ == "__main__":
    run_benchmarks()
    generate_leaderboard()