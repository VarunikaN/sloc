import os, sys, random, time, torch, argparse, pandas as pd
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

# --- Environment Fixes & Imports ---
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

# Adjust paths for your environment
repo_path = "/kaggle/working/sloc"
src_path = os.path.join(repo_path, "src")
if src_path not in sys.path: sys.path.insert(0, src_path)

from models import ModelEnv
from sloc import SlocExplanationCreator, AutoProbSlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss

# --- Helper Functions ---
def get_voc_val_images(voc_root):
    val_set_file = os.path.join(voc_root, 'ImageSets/Main/val.txt')
    jpeg_dir = os.path.join(voc_root, 'JPEGImages')
    if not os.path.exists(val_set_file): raise FileNotFoundError(f"VOC val.txt not found at {val_set_file}")
    with open(val_set_file, 'r') as f: image_ids = [line.strip() for line in f.readlines()]
    return [os.path.join(jpeg_dir, f"{img_id}.jpg") for img_id in image_ids]

def save_visual_result(img_pil, sal_tensor, filename):
    img_resized = img_pil.resize((224, 224))
    if isinstance(sal_tensor, np.ndarray): 
        sal_tensor = torch.from_numpy(sal_tensor.copy())
    sal_norm = (sal_tensor - sal_tensor.min()) / (sal_tensor.max() - sal_tensor.min() + 1e-8)
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Result for {os.path.basename(filename)}", fontsize=16)
    ax[0].imshow(img_resized); ax[0].set_title("Original"); ax[0].axis('off')
    ax[1].imshow(sal_norm, cmap='jet'); ax[1].set_title("Saliency"); ax[1].axis('off')
    ax[2].imshow(img_resized)
    ax[2].imshow(sal_norm, cmap='jet', alpha=0.5); ax[2].set_title("Overlay"); ax[2].axis('off')
    plt.savefig(filename); plt.close(fig)

# --- 1. Metric Evaluator (Base Paper Logic) ---
class SLOCPaperEvaluator:
    def __init__(self, model, device):
        self.model, self.device = model, device
        self.blur = T.GaussianBlur(kernel_size=11, sigma=5.0)

    def run(self, img_tensor, sal_map, target_idx, steps=50):
        self.model.eval()
        h, w = img_tensor.shape[-2:]
        if isinstance(sal_map, np.ndarray): sal_map = np.ascontiguousarray(sal_map)
        sal_flatten = sal_map.flatten()
        idx_desc = np.argsort(sal_flatten)[::-1].copy() 
        idx_asc = np.argsort(sal_flatten).copy() # Least important first
        
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
                top_idx, bot_idx = idx_desc[:n], idx_asc[:n]
                
                # Faithfulness Protocol Calibration
                img_ins = black_flat.clone(); img_ins[:, :, top_idx] = img_flat[:, :, top_idx]
                img_del = img_flat.clone(); img_del[:, :, top_idx] = blur_flat[:, :, top_idx]
                
                # POS: Inverse of Deletion (Keeping only most important)
                img_pos = blur_flat.clone(); img_pos[:, :, top_idx] = img_flat[:, :, top_idx]
                
                # NEG: Keeping everything EXCEPT the most important (Inverse of Insertion)
                img_neg = img_flat.clone(); img_neg[:, :, top_idx] = blur_flat[:, :, top_idx]

                for k, img in zip(["ins", "del", "neg", "pos"], [img_ins, img_del, img_neg, img_pos]):
                    out = torch.softmax(self.model(img.view(1, 3, h, w)), dim=1)
                    prob = out[0, target_idx].item()
                    curves[k].append(prob)
                    if k == "ins":
                        curves["aic"].append(1.0 if torch.argmax(out) == target_idx else 0.0)
                        curves["sic"].append(prob)

        auc = {k: np.trapezoid(v, dx=1/steps) for k, v in curves.items()}
        return {"DEL": auc["del"], "INS": auc["ins"], "IDD": auc["ins"] - auc["del"], 
                "POS": auc["pos"], "NEG": auc["neg"], "NPD": auc["neg"] - auc["pos"], 
                "AIC": auc["aic"], "SIC": auc["sic"]}

# --- 2. SLOC_m Creator (Epoch Logging) ---
class SlocM_Creator(SlocExplanationCreator):
    def explain(self, me, inp, catidx, image_name="unknown"):
        # CSV 1: Epoch Logs for this backbone
        epoch_csv = f"epoch_logs_{me.arch}_voc.csv"
        
        data = self.generate_data(me, inp, catidx)
        initial = (torch.randn(me.shape[0], me.shape[1]) * 0.2 + 3).to(me.device)
        mexp = MaskedExplanationSum(initial_value=initial, H=me.shape[0], W=me.shape[1]).to(me.device)
        optimizer = optim.Adam(mexp.parameters(), lr=0.1); tv = TotalVariationLoss()
        
        epoch_logs = []
        for epoch in range(501):
            optimizer.zero_grad()
            output = mexp(data.all_masks)
            comp_loss = ((output - data.all_pred)**2).mean()
            tv_loss = 0.05 * tv(mexp.explanation)
            mag_loss = 0.01 * mexp.explanation.abs().mean()
            loss = comp_loss + tv_loss + mag_loss
            loss.backward(); optimizer.step()

            if epoch % 100 == 0 or epoch == 500:
                epoch_logs.append({
                    'Image': image_name, 'Epoch': epoch, 'Loss': loss.item(),
                    'CompLoss': comp_loss.item(), 'TVLoss': tv_loss.item(), 'MagLoss': mag_loss.item()
                })
        
        pd.DataFrame(epoch_logs).to_csv(epoch_csv, mode='a', header=not os.path.exists(epoch_csv), index=False)
        return mexp.explanation.detach().cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description="PASCAL VOC Benchmark")
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC', 'SLOC_xp', 'SLOC_m'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--num_images', type=int, default=1000)
    args = parser.parse_args()

    me = ModelEnv(args.model)
    p = 0.3 if 'vit' in args.model.lower() else 0.6
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'pprob': [p]*3}

    if args.variant == 'SLOC_m': creator = SlocM_Creator(**config)
    else: creator = SlocExplanationCreator(**config)

    VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012"
    images = get_voc_val_images(VOC_ROOT)
    selected_images = random.sample(images, min(args.num_images, len(images)))
    
    # CSV 2: Metrics Results
    output_csv = f"sloc_results_{args.model}_voc.csv"
    visuals_dir = f"visuals_{args.model}"
    os.makedirs(visuals_dir, exist_ok=True)

    start_time = time.time()
    for i, path in enumerate(selected_images):
        name = os.path.basename(path)
        try:
            img_pil, inp = me.get_image_ext(path)
            target = torch.argmax(me.model(inp)).item()
            
            # 1. Run Explanation + Epoch Logging (CSV 1 generated inside)
            sal_numpy = creator.explain(me, inp, target, image_name=name)
            
            # 2. Metrics Calculation (CSV 2 generated here)
            m = SLOCPaperEvaluator(me.model, me.device).run(inp, sal_numpy, target)
            m['Image'] = name
            pd.DataFrame([m]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
            
            print(f"[{i+1}/{len(selected_images)}] {name} | IDD: {m['IDD']:.4f} | NPD: {m['NPD']:.4f}")

            # Visuals folder saving every 10 images
            if (i + 1) % 10 == 0:
                visual_path = os.path.join(visuals_dir, f"{name}.png")
                save_visual_result(img_pil, sal_numpy, visual_path)

        except Exception as e:
            print(f"FAILED for {name}: {e}")

    print(f"Benchmark Complete. Total Time: {(time.time()-start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()