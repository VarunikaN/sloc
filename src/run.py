import os, sys, random, time, torch, argparse, pandas as pd
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

# ==============================================================================
#                            CONFIGURATION & SETUP
# ==============================================================================
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

repo_path = "/kaggle/working/sloc"
src_path = os.path.join(repo_path, "src")
if src_path not in sys.path: sys.path.insert(0, src_path)

# --- Imports ---
from models import ModelEnv
from sloc import SlocExplanationCreator, AutoProbSlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss
from visutils import showsal

# --- Helper Functions (Moved to Top Level for Scope Fix) ---
def get_voc_val_images(voc_root):
    val_set_file = os.path.join(voc_root, 'ImageSets/Main/val.txt')
    jpeg_dir = os.path.join(voc_root, 'JPEGImages')
    if not os.path.exists(val_set_file): raise FileNotFoundError(f"VOC val.txt not found at {val_set_file}")
    with open(val_set_file, 'r') as f: image_ids = [line.strip() for line in f.readlines()]
    return [os.path.join(jpeg_dir, f"{img_id}.jpg") for img_id in image_ids]

def save_visual_result(img_pil, sal_tensor, filename):
    img_resized = img_pil.resize((224, 224))
    sal_norm = (sal_tensor - sal_tensor.min()) / (sal_tensor.max() - sal_tensor.min() + 1e-8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5)); fig.suptitle(f"Result for {os.path.basename(filename)}", fontsize=16)
    ax[0].imshow(img_resized); ax[0].set_title("Original"); ax[0].axis('off')
    ax[1].imshow(sal_norm, cmap='jet'); ax[1].set_title("Saliency"); ax[1].axis('off')
    ax[2].imshow(img_resized); ax[2].imshow(sal_norm, cmap='jet', alpha=0.5); ax[2].set_title("Overlay"); ax[2].axis('off')
    plt.savefig(filename); plt.close(fig)

# --- 1. Metric Evaluator (All 8 Metrics) ---
class SLOCPaperEvaluator:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.blur = T.GaussianBlur(kernel_size=11, sigma=5.0)

    def run(self, img_tensor, sal_map, target_idx, steps=50):
        self.model.eval()
        h, w = img_tensor.shape[-2:]
        if isinstance(sal_map, np.ndarray): sal_map = np.ascontiguousarray(sal_map)
        sal_flatten = sal_map.flatten()
        idx_desc = np.argsort(sal_flatten)[::-1]; idx_asc = np.argsort(sal_flatten)
        
        black_base = torch.zeros_like(img_tensor).to(self.device)
        blur_base = self.blur(img_tensor).to(self.device)
        
        curves = {k: [] for k in ["ins", "del", "pos", "neg", "aic", "sic"]}
        step_size = len(idx_desc) // steps

        with torch.no_grad():
            img_flat = img_tensor.view(1, 3, -1).clone().contiguous()
            blur_flat = blur_base.view(1, 3, -1).clone().contiguous()
            black_flat = black_base.view(1, 3, -1).clone().contiguous()

            for i in range(steps + 1):
                n = min(i * step_size, len(idx_desc))
                top_idx = idx_desc[:n]; bot_idx = idx_asc[:n]

                img_ins = black_flat.clone(); img_ins[:, :, top_idx] = img_flat[:, :, top_idx]
                img_del = img_flat.clone(); img_del[:, :, top_idx] = blur_flat[:, :, top_idx]
                img_neg = img_flat.clone(); img_neg[:, :, bot_idx] = blur_flat[:, :, bot_idx]

                for k, img in zip(["ins", "del", "neg", "pos"], [img_ins, img_del, img_neg, img_ins]):
                    out = torch.softmax(self.model(img.view(1, 3, h, w)), dim=1)
                    prob = out[0, target_idx].item()
                    curves[k].append(prob)
                    if k == "ins":
                        curves["aic"].append(1.0 if torch.argmax(out) == target_idx else 0.0)
                        curves["sic"].append(prob)

        auc = {k: np.trapz(v, dx=1/steps) for k, v in curves.items()}
        return {"DEL": auc["del"], "INS": auc["ins"], "IDD": auc["ins"] - auc["del"], "POS": auc["pos"], "NEG": auc["neg"], "NPD": auc["neg"] - auc["pos"], "AIC": auc["aic"], "SIC": auc["sic"]}

# --- 2. SLOC_m Creator (Monitoring IDD) ---
class SlocM_Creator(SlocExplanationCreator):
    def explain(self, me, inp, catidx):
        data = self.generate_data(me, inp, catidx)
        initial = (torch.randn(me.shape[0], me.shape[1]) * 0.2 + 3).to(me.device)
        evaluator = SLOCPaperEvaluator(me.model, me.device)
        state = {'best_idd': -float('inf'), 'best_sal': None}
        mexp = MaskedExplanationSum(initial_value=initial, H=me.shape[0], W=me.shape[1]).to(me.device)
        optimizer = optim.Adam(mexp.parameters(), lr=0.1); tv = TotalVariationLoss()
        C_TV = 0.05; C_MAG = 0.01

        for epoch in range(501):
            optimizer.zero_grad()
            output = mexp(data.all_masks)
            loss = ((output - data.all_pred)**2).mean() + C_TV * tv(mexp.explanation) + C_MAG * mexp.explanation.abs().mean()
            loss.backward(); optimizer.step()

            if epoch % 50 == 0:
                cur_sal = np.ascontiguousarray(mexp.explanation.detach().cpu().numpy())
                metrics = evaluator.run(inp, cur_sal, catidx, steps=10)
                current_idd = metrics["IDD"]
                if current_idd > state['best_idd']:
                    state['best_idd'] = current_idd; state['best_sal'] = cur_sal.copy()
                    
        return state['best_sal']

# --- 3. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Run the SLOC benchmark on the PASCAL VOC 2012 dataset.")
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC', 'SLOC_xp', 'SLOC_m'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_images', type=int, default=1000)
    
    args = parser.parse_args()

    # Load Model Environment
    me = ModelEnv(args.model)
    print(f"GPU CHECK: Model is running on device: {me.device}")
    
    # Paper Calibrations
    p = 0.3 if 'vit' in args.model.lower() else 0.6
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'pprob': [p]*3}

    # Creator Selection
    if args.variant == 'SLOC_m': creator = SlocM_Creator(**config)
    elif args.variant == 'SLOC_xp': creator = SlocExplanationCreator(**config)
    else: creator = AutoProbSlocExplanationCreator(**config)

    # Dataset: PASCAL VOC 2012
    VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012"
    images = get_voc_val_images(VOC_ROOT)
    
    selected_images = random.sample(images, min(args.num_images, len(images)))
    NUM_IMAGES = len(selected_images); output_csv = f"sloc_{args.variant.lower()}_{args.model}_voc{NUM_IMAGES}_results.csv"
    if os.path.exists(output_csv): os.remove(output_csv)

    print(f"Starting PASCAL VOC Benchmark ({args.variant} on {args.model}) for {NUM_IMAGES} images.")
    start_time = time.time(); results_list = []

    for i, path in enumerate(selected_images):
        name = os.path.basename(path)
        eta = f" | ETA: {(time.time() - start_time) / i * (NUM_IMAGES - i) / 3600:.2f} hours" if i > 0 else ""

        print(f"\n[{i+1}/{NUM_IMAGES}] Processing {name}...{eta}")
        
        try:
            img_pil, inp = me.get_image_ext(path); target = torch.argmax(me.model(inp)).item()
            sal_numpy = creator.explain(me, inp, target)
            m = SLOCPaperEvaluator(me.model, me.device).run(inp, sal_numpy, target, steps=50)
            m['Image'] = name; results_list.append(m)
            
            # Incremental Save
            df_temp = pd.DataFrame([m]); header = not os.path.exists(output_csv)
            df_temp.to_csv(output_csv, mode='a', header=header, index=False)
            
            print(f"   IDD: {m['IDD']:.4f}, AIC: {m['AIC']:.4f}, NPD: {m['NPD']:.4f}")

            if (i+1) % 100 == 0:
                visual_filename = f"visuals/res_{args.variant}_{args.model}_{i+1}_{name}.png"
                save_visual_result(img_pil, torch.tensor(sal_numpy), visual_filename)
                print(f"   Saved visual result to {visual_filename}")

        except Exception as e:
            print(f"   !!! FAILED for {name}: {e}"); with open("failed_images.log", "a") as log_file: log_file.write(f"{name}: {e}\n")

    # Final Report
    total_time = time.time() - start_time
    print("\n" + "="*80); print(f"BENCHMARK COMPLETED for {args.variant} in {total_time/3600:.2f} hours"); print("="*80)
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv); print(f"FINAL AVERAGE SCORES ({args.variant}, {args.model}, {NUM_IMAGES} images):")
        print(df.mean(numeric_only=True).round(4)); print("="*80)

if __name__ == "__main__":
    main()