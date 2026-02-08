import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
import warnings
import time
import matplotlib.pyplot as plt
import argparse

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================

# --- Pre-Flight Check & Setup ---
def setup_environment():
    """Ensures the environment is ready for the benchmark."""
    if 'src' not in os.getcwd():
        if os.path.exists('sloc/src'): os.chdir('sloc')
        else: raise FileNotFoundError("CRITICAL ERROR: Please run this script from the root of the 'sloc' repository.")
    
    models_path = 'src/models.py'
    try:
        with open(models_path, 'r') as f: content = f.read()
        if 'torchray' in content and '# import timm, torchray' not in content:
            content = content.replace('import timm, torchray', '# import timm, torchray')
            content = content.replace('import torchray.benchmark', '# import torchray.benchmark')
            with open(models_path, 'w') as f: f.write(content)
    except Exception as e: pass # Already patched or error in setup
    
    warnings.filterwarnings("ignore")
    os.makedirs("visuals", exist_ok=True)

setup_environment()

# --- Safe Imports ---
from src.models import ModelEnv
from src.sloc import SlocExplanationCreator, AutoProbSlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss
import torch.optim as optim
import torch.nn.functional as F

# --- 1. Metric Evaluator (All 8 Metrics) ---
class XAIMetrics:
    def __init__(self, model, device): self.model, self.device = model, device
    def auc(self, arr): return np.trapz(arr, dx=1.0 / (len(arr) - 1)) if len(arr) > 1 else 0
    def evaluate_all(self, img, explanation, target_class, steps=50):
        self.model.eval()
        b, c, h, w = img.shape; n_pixels = h * w
        exp_flat = explanation.flatten(); sorted_indices = torch.argsort(exp_flat, descending=True)
        black_baseline = torch.zeros_like(img).to(self.device)
        blur_baseline = torch.nn.functional.avg_pool2d(img, kernel_size=11, stride=1, padding=5)
        orig_flat, black_flat, blur_flat = img.view(1, 3, -1), black_baseline.view(1, 3, -1), blur_baseline.view(1, 3, -1)
        metrics = {'ins_probs': [], 'del_probs': [], 'pos_probs': [], 'neg_probs': [], 'aic_hits': [], 'sic_sum': []}
        curr_ins, curr_del = blur_baseline.clone().view(1, 3, -1), img.clone().view(1, 3, -1)
        curr_pos, curr_neg = black_baseline.clone().view(1, 3, -1), img.clone().view(1, 3, -1)
        step_size = max(1, n_pixels // steps)
        with torch.no_grad():
            for i in range(steps + 1):
                count = min(i * step_size, n_pixels)
                if count > 0:
                    idx = sorted_indices[:count]
                    curr_ins[:, :, idx], curr_del[:, :, idx] = orig_flat[:, :, idx], blur_flat[:, :, idx]
                    curr_pos[:, :, idx], curr_neg[:, :, idx] = orig_flat[:, :, idx], black_flat[:, :, idx]
                out_ins, out_del = self.model(curr_ins.view(1, 3, h, w)), self.model(curr_del.view(1, 3, h, w))
                out_pos, out_neg = self.model(curr_pos.view(1, 3, h, w)), self.model(curr_neg.view(1, 3, h, w))
                probs_ins = torch.softmax(out_ins, dim=1)
                metrics['ins_probs'].append(probs_ins[0, target_class].item())
                metrics['del_probs'].append(torch.softmax(out_del, dim=1)[0, target_class].item())
                metrics['pos_probs'].append(torch.softmax(out_pos, dim=1)[0, target_class].item())
                metrics['neg_probs'].append(torch.softmax(out_neg, dim=1)[0, target_class].item())
                metrics['aic_hits'].append(1.0 if torch.argmax(out_ins).item() == target_class else 0.0)
                metrics['sic_sum'].append(probs_ins[0, target_class].item())
        results = {"INS": self.auc(metrics['ins_probs']), "DEL": self.auc(metrics['del_probs']), "POS": self.auc(metrics['pos_probs']), "NEG": self.auc(metrics['neg_probs']), "AIC": self.auc(metrics['aic_hits']), "SIC": self.auc(metrics['sic_sum'])}
        results["IDD"], results["NPD"] = results["INS"] - results["DEL"], results["NEG"] - results["POS"]
        return results

# --- 2. SLOC_m Creator (Monitoring/Best-of-Epoch) ---
class SlocM_ExplanationCreator(SlocExplanationCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
    def __call__(self, me, inp, catidx, **kwargs):
        sal = self.explain(me, inp, catidx, **kwargs)
        return {self.description(): sal}
    def explain(self, me, inp, catidx, **kwargs):
        data = self.generate_data(me, inp, catidx)
        initial = (torch.randn(me.shape[0], me.shape[1]) * 0.2 + 3)
        fmdl = me.narrow_model(catidx, with_softmax=True)
        evaluator = XAIMetrics(me.model, me.device)
        state = {'best_idd': -float('inf'), 'best_explanation': None}
        def monitor_callback(explanation_tensor):
            metrics = evaluator.evaluate_all(inp, explanation_tensor, catidx, steps=10)
            current_idd = metrics['IDD']
            if current_idd > state['best_idd']:
                state['best_idd'] = current_idd
                state['best_explanation'] = explanation_tensor.clone()
        final_explanation = self.optimize_explanation_with_monitoring(fmdl, inp, initial, data, callback=monitor_callback, **self.kwargs)
        return state['best_explanation'] if state['best_explanation'] is not None else final_explanation
    def optimize_explanation_with_monitoring(self, fmdl, inp, initial_explanation, data, callback=None, **kwargs):
        shape = inp.shape[-2:]
        mexp = MaskedExplanationSum(initial_value=initial_explanation, H=shape[0], W=shape[1]).to(data.all_masks.device)
        optimizer = optim.Adam(mexp.parameters(), lr=self.lr)
        tv = TotalVariationLoss()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = mexp(data.all_masks); comp_loss = ((output - data.all_pred) ** 2).mean()
            tv_loss = tv(mexp.explanation); mag_loss = mexp.explanation.abs().mean()
            total_loss = comp_loss + self.c_tv * tv_loss + self.c_magnitude * mag_loss
            total_loss.backward(); optimizer.step()
            if callback and (epoch % 50 == 0 and epoch > 0):
                callback(mexp.explanation.detach().cpu())
        return mexp.explanation.detach()

# --- 3. Helper Functions ---
def get_voc_val_images(voc_root):
    val_set_file = os.path.join(voc_root, 'ImageSets/Main/val.txt')
    jpeg_dir = os.path.join(voc_root, 'JPEGImages')
    if not os.path.exists(val_set_file): raise FileNotFoundError(f"Val set file not found at {val_set_file}")
    with open(val_set_file, 'r') as f: image_ids = [line.strip() for line in f.readlines()]
    image_paths = [os.path.join(jpeg_dir, f"{img_id}.jpg") for img_id in image_ids]
    return image_paths

def save_visual_result(img_pil, sal_tensor, filename):
    img_resized = img_pil.resize((224, 224))
    sal_norm = (sal_tensor - sal_tensor.min()) / (sal_tensor.max() - sal_tensor.min() + 1e-8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5)); fig.suptitle(f"Result for {os.path.basename(filename)}", fontsize=16)
    ax[0].imshow(img_resized); ax[0].set_title("Original"); ax[0].axis('off')
    ax[1].imshow(sal_norm, cmap='jet'); ax[1].set_title("Saliency"); ax[1].axis('off')
    ax[2].imshow(img_resized); ax[2].imshow(sal_norm, cmap='jet', alpha=0.5); ax[2].set_title("Overlay"); ax[2].axis('off')
    plt.savefig(filename); plt.close(fig)

# --- 4. Main Execution Block ---
def main(args):
    # Determine Model-Specific Probability (p) for SLOC_xp and SLOC_m
    if 'vit' in args.model.lower():
        pprob_value = 0.3 # ViT uses a lower probability for better results
    else:
        pprob_value = 0.6 # ResNet50 and DenseNet201 use 0.6
        
    pprob_config = [pprob_value] * 3 
    
    # Setup models and evaluators
    me = ModelEnv(args.model)
    print(f"GPU CHECK: Model is running on device: {me.device}")
    evaluator = XAIMetrics(me.model, me.device)

    # Paper's recommended high-quality settings
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'epochs': 500, 'c_tv': 0.05, 'c_magnitude': 0.01, 'lr': 0.1}

    # Instantiate the correct creator based on the chosen variant
    if args.variant == 'SLOC_xp':
        print(f"Initializing SLOC_xp (Fixed Probability p={pprob_value})")
        creator = SlocExplanationCreator(pprob=pprob_config, **config)
    elif args.variant == 'SLOC':
        print("Initializing SLOC (Auto-Tuning Probability)")
        # Note: AutoProbSlocCreator will run its own tuning, ignoring the pprob config here.
        creator = AutoProbSlocExplanationCreator(**config) 
    elif args.variant == 'SLOC_m':
        print(f"Initializing SLOC_m (Monitoring, Fixed Probability p={pprob_value})")
        creator = SlocM_ExplanationCreator(pprob=pprob_config, **config)

    # Load and sample the dataset
    VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012"
    all_images = get_voc_val_images(VOC_ROOT)
    
    if args.num_images > 0:
        selected_images = random.sample(all_images, min(args.num_images, len(all_images)))
    else:
        selected_images = all_images # Use all images
    
    NUM_IMAGES = len(selected_images)
    output_csv = f"sloc_{args.variant.lower()}_{args.model}_voc{NUM_IMAGES}_results.csv"
    if os.path.exists(output_csv): os.remove(output_csv)

    print(f"Starting PASCAL VOC Benchmark ({args.variant} on {args.model}) for {NUM_IMAGES} images.")
    print("This may take several hours. Progress is saved incrementally.")
    start_time = time.time()

    # Main Loop
    for idx, img_path in enumerate(selected_images):
        name = os.path.basename(img_path)
        image_start_time = time.time()
        eta = ''
        if idx > 0:
            avg_time = (time.time() - start_time) / idx
            eta_seconds = avg_time * (NUM_IMAGES - idx)
            eta = f" | ETA: {eta_seconds/3600:.2f} hours"
        print(f"\n[{idx+1}/{NUM_IMAGES}] Processing {name}...{eta}")
        
        try:
            img_pil, inp = me.get_image_ext(img_path)
            logits = me.model(inp); target = torch.argmax(logits).item()
            
            sal_dict = creator(me, inp, target)
            sal = list(sal_dict.values())[0].detach().cpu().squeeze()
            
            m = evaluator.evaluate_all(inp, sal, target, steps=50)
            m['Image'] = name
            
            # Incremental Save
            df_temp = pd.DataFrame([m])
            header = not os.path.exists(output_csv)
            df_temp.to_csv(output_csv, mode='a', header=header, index=False)
            
            image_time = time.time() - image_start_time
            print(f"   Finished in {image_time:.2f}s. IDD: {m['IDD']:.3f}")
            
            if (idx + 1) % 100 == 0:
                visual_filename = f"visuals/result_{args.variant}_{args.model}_{idx+1}_{name}.png"
                print(f"   Saving visual result to {visual_filename}")
                save_visual_result(img_pil, sal, visual_filename)
        except Exception as e:
            print(f"   !!! FAILED for {name}: {e}")
            with open("failed_images.log", "a") as log_file: log_file.write(f"{name}: {e}\n")

    # Final Report
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"BENCHMARK COMPLETED for {args.variant} on {args.model} in {total_time/3600:.2f} hours")
    print("="*80)
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        cols = ['Image', 'INS', 'DEL', 'IDD', 'POS', 'NEG', 'NPD', 'AIC', 'SIC']
        df = df[cols]
        print(f"FINAL AVERAGE SCORES ({args.variant}, {args.model}, {NUM_IMAGES} images):")
        print(df.mean(numeric_only=True).round(4))
        print("="*80)
        print(f"Full results saved to {output_csv}")
    else:
        print("No results were generated.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the SLOC benchmark on the PASCAL VOC 2012 dataset.")
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC_xp', 'SLOC', 'SLOC_m'],
                        help="The SLOC variant to run (SLOC_xp: fixed p, SLOC: auto-tune p, SLOC_m: best-of-epoch).")
    parser.add_argument('--model', type=str, default='resnet50',
                        help="The model architecture to use (e.g., 'resnet50', 'densenet201', 'vit_base_patch16_224').")
    parser.add_argument('--num_images', type=int, default=-1,
                        help="Number of images to sample from the dataset. Set to -1 to use all available images (5823).")
    
    args = parser.parse_args()
    main(args)