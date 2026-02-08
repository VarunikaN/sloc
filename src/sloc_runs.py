import os
import glob
import random
import torch
import numpy as np
import pandas as pd
from PIL import Image
import warnings
import time

VARIANT_TO_RUN = 'SLOC_m'

warnings.filterwarnings("ignore")

from src.models import ModelEnv
from src.sloc import SlocExplanationCreator, AutoProbSlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss
from src.visutils import showsal
import torch.optim as optim
import torch.nn.functional as F

# --- 1. Metric Evaluator (All 8 Metrics) ---
class XAIMetrics:
    def __init__(self, model, device): self.model, self.device = model, device
    def auc(self, arr): return np.trapz(arr, dx=1.0 / (len(arr) - 1)) if len(arr) > 1 else 0
    def evaluate_all(self, img, explanation, target_class, steps=20):
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

# --- 2. SLOC_m Creator (with monitoring) ---
class SlocM_ExplanationCreator(SlocExplanationCreator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
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
            return current_idd
        final_explanation = self.optimize_explanation_with_monitoring(fmdl, inp, initial, data, callback=monitor_callback, **self.kwargs)
        # Return the best map found during monitoring
        sal = state['best_explanation'] if state['best_explanation'] is not None else final_explanation
        return {self.description(): sal} # Return as dict to be consistent
    def optimize_explanation_with_monitoring(self, fmdl, inp, initial_explanation, data, callback=None, **kwargs):
        shape = inp.shape[-2:]
        mexp = MaskedExplanationSum(initial_value=initial_explanation, H=shape[0], W=shape[1]).to(data.all_masks.device)
        optimizer = optim.Adam(mexp.parameters(), lr=self.lr)
        tv = TotalVariationLoss()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = mexp(data.all_masks)
            comp_loss = ((output - data.all_pred) ** 2).mean()
            tv_loss = tv(mexp.explanation); mag_loss = mexp.explanation.abs().mean()
            total_loss = comp_loss + self.c_tv * tv_loss + self.c_magnitude * mag_loss
            total_loss.backward(); optimizer.step()
            if callback and (epoch % 50 == 0 and epoch > 0): # Check less frequently to speed up
                callback(mexp.explanation.detach().cpu())
        return mexp.explanation.detach()

# --- 3. Experiment Setup ---
me = ModelEnv('resnet50')
evaluator = XAIMetrics(me.model, me.device)
creator = None

# High-quality settings used for all variants
config = {
    'segsize': [16, 32, 48], 
    'nmasks': [800, 600, 400],
    'epochs': 500, 
    'c_tv': 0.05, 
    'c_magnitude': 0.01, 
    'lr': 0.1
}

# Instantiate the correct creator based on the VARIANT_TO_RUN setting
if VARIANT_TO_RUN == 'SLOC_xp':
    print("Initializing SLOC_xp (Fixed Probability)")
    creator = SlocExplanationCreator(pprob=[0.6, 0.6, 0.6], **config)
elif VARIANT_TO_RUN == 'SLOC':
    print("Initializing SLOC (Auto-Tuning Probability)")
    creator = AutoProbSlocExplanationCreator(**config)
elif VARIANT_TO_RUN == 'SLOC_m':
    print("Initializing SLOC_m (Best-of-Epoch Monitoring)")
    creator = SlocM_ExplanationCreator(pprob=[0.6, 0.6, 0.6], **config)
else:
    raise ValueError("Invalid VARIANT_TO_RUN. Choose 'SLOC_xp', 'SLOC', or 'SLOC_m'.")

# --- 4. Data Loading and Execution ---
VOC_PATH = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012/JPEGImages"
all_voc_images = glob.glob(os.path.join(VOC_PATH, "*.jpg"))
NUM_IMAGES = 200
selected_images = random.sample(all_voc_images, min(NUM_IMAGES, len(all_voc_images)))
results_log = []
output_csv = f"sloc_{VARIANT_TO_RUN.lower()}_voc200_results.csv"

if os.path.exists(output_csv): os.remove(output_csv)

print(f"Starting Benchmark for VARIANT: {VARIANT_TO_RUN} on {len(selected_images)} images.")
print(f"Results will be saved incrementally to {output_csv}")
start_time = time.time()

for idx, img_path in enumerate(selected_images):
    name = os.path.basename(img_path)
    image_start_time = time.time()
    print(f"\n[{idx+1}/{NUM_IMAGES}] Processing {name}...")
    
    try:
        img_pil, inp = me.get_image_ext(img_path)
        logits = me.model(inp)
        target = torch.argmax(logits).item()
        
        # This one line runs the correct logic for the chosen variant
        sal_dict = creator(me, inp, target)
        sal = list(sal_dict.values())[0].detach().cpu().squeeze()

        m = evaluator.evaluate_all(inp, sal, target, steps=50)
        m['Image'] = name
        results_log.append(m)
        
        # Incremental Save
        df_temp = pd.DataFrame([m])
        header = not os.path.exists(output_csv)
        df_temp.to_csv(output_csv, mode='a', header=header, index=False)
        
        image_time = time.time() - image_start_time
        print(f"   Finished in {image_time:.2f}s. Stats -> IDD: {m['IDD']:.3f}, AIC: {m['AIC']:.3f}")
        
    except Exception as e:
        print(f"   !!! FAILED for {name}: {e}")

# --- 5. Final Report ---
total_time = time.time() - start_time
print("\n" + "="*80)
print(f"BENCHMARK COMPLETED for {VARIANT_TO_RUN} in {total_time/60:.2f} minutes")
print("="*80)

if results_log:
    df = pd.read_csv(output_csv) # Read the final CSV for the full report
    cols = ['Image', 'INS', 'DEL', 'IDD', 'POS', 'NEG', 'NPD', 'AIC', 'SIC']
    df = df[cols]
    print("AVERAGE SCORES:")
    print(df.mean(numeric_only=True).round(4))
    print("="*80)
    print(f"Full results saved to {output_csv}")
else:
    print("No results were generated.")