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
        
        if isinstance(sal_map, np.ndarray):
            sal_map = np.ascontiguousarray(sal_map)
            
        sal_flatten = sal_map.flatten()
        idx_desc = np.argsort(sal_flatten)[::-1].copy() 
        idx_asc = np.argsort(sal_flatten).copy()
        
        black_base = torch.zeros_like(img_tensor).to(self.device)
        blur_base = self.blur(img_tensor).to(self.device)
        
        curves = {k: [] for k in ["ins", "del", "pos", "neg", "aic", "sic"]}
        step_size = len(idx_desc) // steps

        with torch.no_grad():
            img_flat = img_tensor.view(1, 3, -1).contiguous()
            blur_flat = blur_base.view(1, 3, -1).contiguous()
            black_flat = black_base.view(1, 3, -1).contiguous()

            for i in range(steps + 1):
                n = min(i * step_size, len(idx_desc))
                top_idx = idx_desc[:n]
                bot_idx = idx_asc[:n]

                img_ins = black_flat.clone()
                img_ins[:, :, top_idx] = img_flat[:, :, top_idx]
                img_del = img_flat.clone()
                img_del[:, :, top_idx] = blur_flat[:, :, top_idx]
                img_neg = img_flat.clone()
                img_neg[:, :, bot_idx] = blur_flat[:, :, bot_idx]

                for k, img in zip(["ins", "del", "neg", "pos"], [img_ins, img_del, img_neg, img_ins]):
                    out = torch.softmax(self.model(img.view(1, 3, h, w)), dim=1)
                    prob = out[0, target_idx].item()
                    curves[k].append(prob)
                    if k == "ins":
                        curves["aic"].append(1.0 if torch.argmax(out) == target_idx else 0.0)
                        curves["sic"].append(prob)

        # Calculate AUCs
        # Use np.trapezoid to avoid the DeprecationWarning in newer NumPy versions
        auc = {k: np.trapezoid(v, dx=1/steps) for k, v in curves.items()}
        
        # FIX: Explicitly map these to the keys the rest of your script expects
        return {
            "DEL": auc["del"], 
            "INS": auc["ins"], 
            "IDD": auc["ins"] - auc["del"], 
            "POS": auc["pos"], 
            "NEG": auc["neg"], 
            "NPD": auc["neg"] - auc["pos"], 
            "AIC": auc["aic"], 
            "SIC": auc["sic"]
        }

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
            output = mexp(data.all_masks); loss = ((output - data.all_pred)**2).mean() + C_TV * tv(mexp.explanation) + C_MAG * mexp.explanation.abs().mean()
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
    parser = argparse.ArgumentParser(description="Run SLOC Benchmark with RSNA/VOC support.")
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC', 'SLOC_xp', 'SLOC_m'])
    parser.add_argument('--model', type=str, default='resnet50', help="resnet50, densenet201, or vit_base_patch16_224")
    parser.add_argument('--dataset', type=str, default='voc', choices=['voc', 'rsna'])
    parser.add_argument('--num_images', type=int, default=1000)
    parser.add_argument('--resume', action='store_true', help="Skip images already present in results CSV")
    
    args = parser.parse_args()

    # 1. Load Model Environment
    me = ModelEnv(args.model)
    print(f"GPU CHECK: Model {args.model} is running on: {me.device}")
    
    # 2. Paper Calibrations for Architecture
    # ViT density p=0.3, CNN density p=0.6 (Section 4.1)
    if 'vit' in args.model.lower(): 
        p = 0.3
    else: 
        p = 0.6
        
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'pprob': [p]*3}

    # 3. Variant Selection
    if args.variant == 'SLOC_m': 
        creator = SlocM_Creator(**config)
    elif args.variant == 'SLOC_xp': 
        creator = SlocExplanationCreator(**config)
    else: 
        creator = AutoProbSlocExplanationCreator(**config)

    if args.dataset == 'rsna':
        RSNA_ROOT = "/kaggle/input/rsna-pneumonia-detection-challenge"
        source = RSNASource(RSNA_ROOT)
        all_images = list(source.get_all_images().values())
    else:
        VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012"
        all_images = get_voc_val_images(VOC_ROOT) 
    
    # 5. Sampling and Resume Logic
    if args.num_images > 0:
        selected_images = random.sample(all_images, min(args.num_images, len(all_images)))
    else:
        selected_images = all_images
        
    output_csv = f"sloc_{args.variant.lower()}_{args.model}_{args.dataset}_results.csv"
    
    processed_images = set()
    if args.resume and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            processed_images = set(existing_df['Image'].astype(str).tolist())
            print(f"Resuming: Skipping {len(processed_images)} images already in {output_csv}")
        except Exception as e:
            print(f"Resume failed (CSV empty or corrupted): {e}")

    # 6. Execution Loop
    print(f"Starting Benchmark: {args.variant} | {args.model} | {args.dataset}")
    start_time = time.time()
    results_list = []

    for i, info in enumerate(selected_images):
        # info is an ImageInfo object (voc) or patient record (rsna)
        image_name = info.name if hasattr(info, 'name') else os.path.basename(info)
        
        if image_name in processed_images:
            continue

        try:
            # Handle RSNA DICOM vs VOC JPEG
            if args.dataset == 'rsna':
                img_pil = load_rsna_as_pil(info.path)
                # Manually apply model transform since me.get_image_ext is for JPEGs
                inp = me.get_transform()(img_pil).unsqueeze(0).to(me.device)
            else:
                img_pil, inp = me.get_image_ext(info.path)
            
            target = torch.argmax(me.model(inp)).item()
            
            # Generate SLOC explanation
            sal_numpy = creator.explain(me, inp, target)
            
            # Evaluate using the 100% Paper Evaluator (8 Metrics)
            m = SLOCPaperEvaluator(me.model, me.device).run(inp, sal_numpy, target, steps=50)
            m['Image'] = image_name
            print(f"   [RESULT] IDD ↑: {m['IDD']:.4f} | NPD ↑: {m['NPD']:.4f} | AIC ↑: {m['AIC']:.4f} | SIC ↑: {m['SIC']:.4f}")
            results_list.append(m)
            
            # Incremental CSV Save
            pd.DataFrame([m]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            
            print(f"[{i+1}/{len(selected_images)}] {image_name} -> IDD: {m['IDD ↑']:.4f}")

        except Exception as e:
            print(f"FAILED for {image_name}: {e}")
            with open("error_log.txt", "a") as f: f.write(f"{image_name}: {str(e)}\n")

    print(f"\nBenchmark Complete. Total Time: {(time.time()-start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()