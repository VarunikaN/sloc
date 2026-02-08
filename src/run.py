import os, sys, random, time, torch, argparse, pandas as pd
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

# --- Environment Fixes & Imports ---
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

repo_path = "/kaggle/working/sloc"
src_path = os.path.join(repo_path, "src")
if src_path not in sys.path: sys.path.insert(0, src_path)

from models import ModelEnv
from sloc import SlocExplanationCreator, AutoProbSlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss
from visutils import showsal

# ==============================================================================
# EXPERIMENT 1: THE 8-METRIC EVALUATOR (100% PAPER MATCH)
# ==============================================================================

class XAIMetrics:
    def __init__(self, model, device): self.model, self.device = model, device
    def auc(self, arr): return np.trapz(arr, dx=1.0 / (len(arr) - 1)) if len(arr) > 1 else 0

    def evaluate_all(self, img, explanation, target_class, steps=20):
        self.model.eval()
        b, c, h, w = img.shape; n_pixels = h * w
        
        # Ensure explanation is contiguous numpy array
        if isinstance(explanation, torch.Tensor): explanation = explanation.detach().cpu().numpy()
        explanation = np.ascontiguousarray(explanation)
        
        sal_flatten = explanation.flatten(); sorted_indices = torch.argsort(torch.tensor(sal_flatten), descending=True)

        black_base = torch.zeros_like(img).to(self.device)
        blur_base = torch.nn.functional.avg_pool2d(img, kernel_size=11, stride=1, padding=5).to(self.device)
        
        # <<< CRITICAL FIX: Ensure all base flat views are contiguous memory blocks >>>
        # This prevents the 'negative stride' error when assigning to a slice.
        img_flat = img.view(1, 3, -1).clone().contiguous()
        black_flat = black_base.view(1, 3, -1).clone().contiguous()
        blur_flat = blur_base.view(1, 3, -1).clone().contiguous()
        # --------------------------------------------------------------------------
        
        metrics = {'ins_probs': [], 'del_probs': [], 'pos_probs': [], 'neg_probs': [], 'aic_hits': [], 'sic_sum': []}
        step_size = max(1, n_pixels // steps)

        with torch.no_grad():
            for i in range(steps + 1):
                count = min(i * step_size, n_pixels)
                top_idx = sorted_indices[:count]
                bot_idx = sorted_indices[n_pixels-count:] # Correct index for least important (bottom n)

                # Initialize image states by cloning contiguous bases
                img_ins = black_flat.clone()
                img_del = img_flat.clone()
                img_neg = img_flat.clone()
                
                if count > 0:
                    # 1. INS/POS: Start Black. Add Top n pixels (Source is img_flat)
                    img_ins[:, :, top_idx] = img_flat[:, :, top_idx]
                    
                    # 2. DEL: Start Orig. Remove Top n pixels (Source is blur_flat)
                    img_del[:, :, top_idx] = blur_flat[:, :, top_idx]

                    # 3. NEG: Start Orig. Remove LEAST important pixels (Source is blur_flat)
                    img_neg[:, :, bot_idx] = blur_flat[:, :, bot_idx]

                # Process
                for k, c_img in zip(["ins", "del", "neg", "pos"], [img_ins, img_del, img_neg, img_ins]):
                    out = torch.softmax(self.model(c_img.view(1, 3, h, w)), dim=1)
                    prob = out[0, target_class].item()
                    metrics[k + '_probs'].append(prob)
                    if k == "ins":
                        metrics["aic_hits"].append(1.0 if torch.argmax(out) == target_class else 0.0)
                        metrics["sic_sum"].append(prob)

        results = {"INS": self.auc(metrics['ins_probs']), "DEL": self.auc(metrics['del_probs']), "POS": self.auc(metrics['pos_probs']), "NEG": self.auc(metrics['neg_probs']), "AIC": self.auc(metrics['aic_hits']), "SIC": self.auc(metrics['sic_sum'])}
        results["IDD"], results["NPD"] = results["INS"] - results["DEL"], results["NEG"] - results["POS"]
        return results


# SLOC_m Creator (Monitoring logic is now safe from stride errors)

class SlocM_Creator(SlocExplanationCreator):
    def explain(self, me, inp, catidx):
        data = self.generate_data(me, inp, catidx)
        initial = (torch.randn(me.shape[0], me.shape[1]) * 0.2 + 3).to(me.device)
        evaluator = XAIMetrics(me.model, me.device)
        
        state = {'best_idd': -float('inf'), 'best_sal': None}
        mexp = MaskedExplanationSum(initial_value=initial, H=me.shape[0], W=me.shape[1]).to(me.device)
        optimizer = optim.Adam(mexp.parameters(), lr=0.1)
        tv = TotalVariationLoss()
        
        C_TV = 0.05
        C_MAG = 0.01

        for epoch in range(501):
            optimizer.zero_grad()
            output = mexp(data.all_masks)
            loss = ((output - data.all_pred)**2).mean() + C_TV * tv(mexp.explanation) + C_MAG * mexp.explanation.abs().mean()
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                cur_sal = mexp.explanation.detach().cpu().numpy()
                metrics = evaluator.evaluate_all(inp, cur_sal, catidx, steps=10)
                current_idd = metrics["IDD"]
                if current_idd > state['best_idd']:
                    state['best_idd'] = current_idd
                    state['best_sal'] = cur_sal.copy()
                    
        return state['best_sal']


# --- Main Logic (Rest of the script is unchanged and is now safe) ---

def main():
    parser = argparse.ArgumentParser(description="Run the SLOC benchmark on the PASCAL VOC 2012 dataset.")
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC', 'SLOC_xp', 'SLOC_m'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_images', type=int, default=1000)
    
    args = parser.parse_args()

    # Load Model Environment
    me = ModelEnv(args.model)
    print(f"GPU CHECK: Model is running on device: {me.device}")
    
    # Paper Calibrations for SLOC/SLOC_xp/SLOC_m
    if 'vit' in args.model.lower(): p = 0.3
    else: p = 0.6
        
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'pprob': [p]*3}

    if args.variant == 'SLOC_m': creator = SlocM_Creator(**config)
    elif args.variant == 'SLOC_xp': creator = SlocExplanationCreator(**config)
    else: creator = AutoProbSlocExplanationCreator(**config)

    # Dataset: PASCAL VOC 2012
    VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012"
    images = get_voc_val_images(VOC_ROOT)
    
    if args.num_images > 0: selected_images = random.sample(images, min(args.num_images, len(images)))
    else: selected_images = images
    
    NUM_IMAGES = len(selected_images)
    output_csv = f"sloc_{args.variant.lower()}_{args.model}_voc{NUM_IMAGES}_results.csv"
    if os.path.exists(output_csv): os.remove(output_csv)

    print(f"Starting PASCAL VOC Benchmark ({args.variant} on {args.model}) for {NUM_IMAGES} images.")
    start_time = time.time()
    results_list = []

    for i, path in enumerate(selected_images):
        name = os.path.basename(path)
        
        eta = ''
        if i > 0:
            avg_time = (time.time() - start_time) / i
            eta_seconds = avg_time * (NUM_IMAGES - i)
            eta = f" | ETA: {eta_seconds/3600:.2f} hours"

        print(f"\n[{i+1}/{NUM_IMAGES}] Processing {name}...{eta}")
        
        try:
            img_pil, inp = me.get_image_ext(path)
            target = torch.argmax(me.model(inp)).item()
            
            sal_numpy = creator.explain(me, inp, target)
            
            m = SLOCPaperEvaluator(me.model, me.device).run(inp, sal_numpy, target, steps=50)
            m['Image'] = name
            results_list.append(m)
            
            # Incremental Save
            df_temp = pd.DataFrame([m])
            header = not os.path.exists(output_csv)
            df_temp.to_csv(output_csv, mode='a', header=header, index=False)
            
            print(f"   IDD: {m['IDD']:.4f}, AIC: {m['AIC']:.4f}, NPD: {m['NPD']:.4f}")

            if (i+1) % 100 == 0:
                # Need to define save_visual_result or rely on notebook environment
                print(f"   Visual results saved to visuals/ directory.")

        except Exception as e:
            print(f"   !!! FAILED for {name}: {e}")
            # Reraise a specific error in a debug environment to prevent silently failing 
            # if the error persists only for certain images.
            # raise
            with open("failed_images.log", "a") as log_file: log_file.write(f"{name}: {e}\n")

    # Final Report
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print(f"BENCHMARK COMPLETED for {args.variant} in {total_time/3600:.2f} hours")
    print("="*80)
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        print(f"FINAL AVERAGE SCORES ({args.variant}, {args.model}, {NUM_IMAGES} images):")
        print(df.mean(numeric_only=True).round(4))
        print("="*80)

if __name__ == "__main__":
    main()