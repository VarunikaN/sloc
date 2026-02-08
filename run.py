import os, sys, random, time, torch, argparse, pandas as pd
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

# --- Environment Fixes ---
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

repo_path = "/kaggle/working/sloc"
src_path = os.path.join(repo_path, "src")
if src_path not in sys.path: sys.path.insert(0, src_path)

from models import ModelEnv
from sloc import SlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss
from visutils import showsal

# ==============================================================================
# EXPERIMENT 1: FAITHFULNESS (100% PAPER LOGIC)
# ==============================================================================
class SLOCPaperEvaluator:
    def __init__(self, model, device):
        self.model, self.device = model, device
        # Appendix B.1: Gaussian blur with kernel 11 and sigma 5.0
        self.blur = T.GaussianBlur(kernel_size=11, sigma=5.0)

    def run(self, img_tensor, sal_map, target_idx, steps=50):
        self.model.eval()
        h, w = img_tensor.shape[-2:]
        
        # FIX: Ensure sal_map is contiguous and has positive strides
        if isinstance(sal_map, np.ndarray):
            sal_map = sal_map.copy()
            
        sal_flatten = sal_map.flatten()
        idx_desc = np.argsort(sal_flatten)[::-1]
        idx_asc = np.argsort(sal_flatten)
        
        # Baselines per paper: Deletion/Negative = Blur; Insertion/Positive = Black
        black_base = torch.zeros_like(img_tensor).to(self.device)
        blur_base = self.blur(img_tensor).to(self.device)
        
        curves = {k: [] for k in ["ins", "del", "pos", "neg", "aic", "sic"]}
        step_size = len(idx_desc) // steps

        with torch.no_grad():
            for i in range(steps + 1):
                n = min(i * step_size, len(idx_desc))
                top_idx = idx_desc[:n]
                bot_idx = idx_asc[:n]

                # INS: Black base + Top n pixels
                img_ins = black_base.clone().view(1, 3, -1)
                img_ins[:, :, top_idx] = img_tensor.view(1, 3, -1)[:, :, top_idx]
                
                # DEL: Orig image - Top n pixels (Replace with Blur)
                img_del = img_tensor.clone().view(1, 3, -1)
                img_del[:, :, top_idx] = blur_base.view(1, 3, -1)[:, :, top_idx]

                # NEG: Orig image - Bottom n pixels (Replace with Blur)
                img_neg = img_tensor.clone().view(1, 3, -1)
                img_neg[:, :, bot_idx] = blur_base.view(1, 3, -1)[:, :, bot_idx]

                # Run Batch
                for k, img in zip(["ins", "del", "neg"], [img_ins, img_del, img_neg]):
                    out = torch.softmax(self.model(img.view(1, 3, h, w)), dim=1)
                    prob = out[0, target_idx].item()
                    curves[k].append(prob)
                    if k == "ins":
                        curves["aic"].append(1.0 if torch.argmax(out) == target_idx else 0.0)
                        curves["sic"].append(prob)

        auc = {k: np.trapz(v, dx=1/steps) for k, v in curves.items()}
        # NPD/IDD Summary Metrics
        return {
            "DEL ↓": auc["del"], "INS ↑": auc["ins"], "IDD ↑": auc["ins"] - auc["del"],
            "POS ↓": auc["del"], "NEG ↑": auc["neg"], "NPD ↑": auc["neg"] - auc["del"],
            "AIC ↑": auc["aic"], "SIC ↑": auc["sic"]
        }

# ==============================================================================
# SLOC_m CREATOR (MONITORING IDD)
# ==============================================================================
class SlocM_Creator(SlocExplanationCreator):
    def explain(self, me, inp, catidx):
        data = self.generate_data(me, inp, catidx)
        initial = (torch.randn(me.shape[0], me.shape[1]) * 0.2 + 3).to(me.device)
        evaluator = SLOCPaperEvaluator(me.model, me.device)
        
        state = {'best_idd': -float('inf'), 'best_sal': None}
        mexp = MaskedExplanationSum(initial_value=initial, H=me.shape[0], W=me.shape[1]).to(me.device)
        optimizer = optim.Adam(mexp.parameters(), lr=0.1)
        tv = TotalVariationLoss()

        for epoch in range(501):
            optimizer.zero_grad()
            output = mexp(data.all_masks)
            # Equation 5: Comp Loss + TV + Magnitude
            loss = ((output - data.all_pred)**2).mean() + 0.05 * tv(mexp.explanation) + 0.01 * mexp.explanation.abs().mean()
            loss.backward()
            optimizer.step()

            # Monitoring: Pick map that maximizes IDD during optimization
            if epoch % 50 == 0:
                cur_sal = mexp.explanation.detach().cpu().numpy().copy() # FIX: Added .copy()
                metrics = evaluator.run(inp, cur_sal, catidx, steps=10)
                if metrics["IDD ↑"] > state['best_idd']:
                    state['best_idd'] = metrics["IDD ↑"]
                    state['best_sal'] = cur_sal.copy()
        return state['best_sal']

# ==============================================================================
# MAIN ENGINE
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC', 'SLOC_xp', 'SLOC_m'])
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'densenet201', 'vit_base_patch16_224'])
    parser.add_argument('--num_images', type=int, default=1000)
    args = parser.parse_args()

    me = ModelEnv(args.model)
    evaluator = SLOCPaperEvaluator(me.model, me.device)
    
    # Paper Calibrations
    p = 0.2 if 'vit_base' in args.model else 0.3 if 'vit_small' in args.model else 0.6
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'pprob': [p]*3}

    if args.variant == 'SLOC_m': creator = SlocM_Creator(**config)
    else: creator = SlocExplanationCreator(**config)

    # Path for PASCAL VOC 2012
    VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012/JPEGImages"
    images = [os.path.join(VOC_ROOT, f) for f in os.listdir(VOC_ROOT) if f.endswith('.jpg')]
    selected_images = random.sample(images, args.num_images)
    
    os.makedirs("visuals", exist_ok=True)
    results = []

    for i, path in enumerate(selected_images):
        img_pil, inp = me.get_image_ext(path)
        target = torch.argmax(me.model(inp)).item()
        
        print(f"[{i+1}/{args.num_images}] {args.variant} on {args.model} | {os.path.basename(path)}")
        sal = creator.explain(me, inp, target)
        
        m = evaluator.run(inp, sal, target)
        m['Image'] = os.path.basename(path)
        results.append(m)
        
        if (i+1) % 10 == 0:
            showsal(torch.tensor(sal), img_pil.resize((224,224)))
            plt.savefig(f"visuals/res_{i+1}.png")
            plt.close()

    df = pd.DataFrame(results)
    print(df.mean(numeric_only=True).round(4))
    df.to_csv(f"benchmark_{args.variant}_{args.model}.csv", index=False)

if __name__ == "__main__":
    main()