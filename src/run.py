import os, sys, random, time, torch, argparse, pandas as pd
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from PIL import Image
from matplotlib import pyplot as plt

# Import the new RSNA logic from your local dataset.py
from dataset import RSNABoneAgeSource, load_boneage_as_pil, ImageInfo

# Existing project imports
from models import ModelEnv
from sloc import SlocExplanationCreator, AutoProbSlocExplanationCreator, MaskedExplanationSum, TotalVariationLoss
from visutils import showsal

# --- Environment Fixes & Imports ---
if not hasattr(np, 'trapz'): np.trapz = np.trapezoid

repo_path = "/kaggle/working/sloc"
src_path = os.path.join(repo_path, "src")
if src_path not in sys.path: sys.path.insert(0, src_path)


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

class EpochLogger:
    def __init__(self, model_name, dataset_name):
        self.filename = f"epoch_logs_{model_name}_{dataset_name}.csv"
        self.data = []

    def log(self, image_id, epoch, total_loss, comp_loss, tv_loss, mag_loss):
        self.data.append({
            'ImageID': image_id,
            'Epoch': epoch,
            'TotalLoss': total_loss,
            'CompLoss': comp_loss,
            'TVLoss': tv_loss,
            'MagLoss': mag_loss
        })

    def save_to_disk(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, mode='a', header=not os.path.exists(self.filename), index=False)
        self.data = []


# --- Helper: Model Output Normalizer ---
class ModelOutputNormalizer:
    """Wraps model to ensure consistent [batch, classes] output across architectures."""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self._is_spatial_output = None
        
    def __call__(self, x):
        out = self.model(x)
        
        # Auto-detect spatial output on first call
        if self._is_spatial_output is None:
            if len(out.shape) == 4:  # [B, H, W, C] - Swin format
                self._is_spatial_output = True
                print(f"DEBUG: Detected spatial output {out.shape}. Will pool to [B, C]")
            elif len(out.shape) == 2:  # [B, C] - Standard format
                self._is_spatial_output = False
                print(f"DEBUG: Detected standard output {out.shape}")
            else:
                raise ValueError(f"Unexpected model output shape: {out.shape}")
        
        # Normalize to [B, C] if needed
        if self._is_spatial_output:
            # Global average pooling over spatial dimensions for Swin
            out = out.mean(dim=[1, 2])  # [B, H, W, C] -> [B, C]
        
        return out
    
    def eval(self):
        self.model.eval()
        return self

      
# --- 1. Metric Evaluator (All 8 Metrics) ---
class SLOCPaperEvaluator:
    def __init__(self, model, device):
        self.model = ModelOutputNormalizer(model, device)
        self.device = device

    def run(self, img_tensor, sal_map, target_idx, steps=50):
        self.model.eval()
        h, w = img_tensor.shape[-2:]
        if isinstance(sal_map, np.ndarray):
            sal_map = np.ascontiguousarray(sal_map)
        
        sal_flatten = sal_map.flatten()
        idx_desc = np.argsort(sal_flatten)[::-1].copy() 
        idx_asc = np.argsort(sal_flatten).copy()
        
        black_base = torch.zeros_like(img_tensor).to(self.device)
        
        curves = {k: [] for k in ["ins", "del", "pos", "neg", "aic", "sic"]}
        step_size = len(idx_desc) // steps

        with torch.no_grad():
            img_flat = img_tensor.view(1, 3, -1).contiguous()
            black_flat = black_base.view(1, 3, -1).contiguous()

            for i in range(steps + 1):
                n = min(i * step_size, len(idx_desc))
                top_idx = idx_desc[:n]
                bot_idx = idx_asc[:n]

                img_ins = black_flat.clone()
                img_ins[:, :, top_idx] = img_flat[:, :, top_idx]
                
                img_del = img_flat.clone()
                img_del[:, :, top_idx] = black_flat[:, :, top_idx]
                
                img_neg = img_flat.clone()
                img_neg[:, :, bot_idx] = black_flat[:, :, bot_idx]

                for k, img in zip(["ins", "del", "neg", "pos"], [img_ins, img_del, img_neg, img_ins]):
                    out = torch.softmax(self.model(img.view(1, 3, h, w)), dim=1)
                    prob = out[0, target_idx].item()
                    curves[k].append(prob)
                    if k == "ins":
                        curves["aic"].append(1.0 if torch.argmax(out) == target_idx else 0.0)
                        curves["sic"].append(prob)

        auc = {k: np.trapezoid(v, dx=1/steps) for k, v in curves.items()}
        return {
            "DEL": auc["del"], "INS": auc["ins"], "IDD": auc["ins"] - auc["del"], 
            "POS": auc["pos"], "NEG": auc["neg"], "NPD": auc["neg"] - auc["pos"], 
            "AIC": auc["aic"], "SIC": auc["sic"]
        }

# --- 2. SLOC_m Creator ---
class SlocM_Creator(SlocExplanationCreator):
    def __init__(self, model_wrapper, **kwargs):
        """Initialize with model wrapper for consistent outputs."""
        super().__init__(**kwargs)
        self.model_wrapper = model_wrapper
        
    def explain(self, me, inp, catidx, image_id="unknown"):
        self.epoch_logger = EpochLogger(me.arch, "rsna_boneage")
        
        # Use wrapped model for data generation
        original_model = me.model
        me.model = self.model_wrapper
        
        data = self.generate_data(me, inp, catidx)
        
        # Restore original model
        me.model = original_model
        
        initial = (torch.randn(me.shape[0], me.shape[1]) * 0.2 + 3)
        final_explanation = self.optimize_with_logs(
            me, inp, initial, data, image_id, catidx
        )
        self.epoch_logger.save_to_disk()
        return final_explanation

    def optimize_with_logs(self, me, inp, initial, data, image_id, catidx):
        # Move initial to GPU
        initial = initial.to(me.device)
        
        # Create and move module to GPU - CRITICAL: This must happen BEFORE optimizer creation
        mexp = MaskedExplanationSum(initial_value=initial, H=me.shape[0], W=me.shape[1])
        mexp = mexp.to(me.device)  # Explicit move to ensure all parameters are on GPU
        
        optimizer = optim.Adam(mexp.parameters(), lr=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        tv = TotalVariationLoss()
        
        # Move all data tensors to GPU FIRST
        masks = data.all_masks.to(me.device)
        all_pred = data.all_pred
        
        # Handle classification vs regression
        if all_pred.numel() == masks.shape[0] * 241:
            # Classification: reshape and extract target class
            targets = all_pred.view(masks.shape[0], 241)[:, catidx].to(me.device)
        else:
            # Regression: use as-is
            targets = all_pred.flatten().to(me.device)

        for epoch in range(501):
            optimizer.zero_grad()
            
            # Forward pass
            output = mexp(masks)
            
            # Losses
            comp_loss = ((output - targets) ** 2).mean()
            tv_loss = 0.05 * tv(mexp.explanation)
            mag_loss = 0.01 * mexp.explanation.abs().mean()
            
            total_loss = comp_loss + tv_loss + mag_loss
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 100 == 0 or epoch == 500:
                self.epoch_logger.log(image_id, epoch, total_loss.item(), 
                                    comp_loss.item(), tv_loss.item(), mag_loss.item())
        
        return mexp.explanation.detach().cpu().numpy()

# --- 3. Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(description="Run SLOC Benchmark on Full RSNA Bone Age Split.")
    parser.add_argument('--variant', type=str, default='SLOC_m', choices=['SLOC', 'SLOC_xp', 'SLOC_m'])
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='rsna_boneage')
    parser.add_argument('--split', type=str, default='training', choices=['training', 'validation'])
    parser.add_argument('--num_images', type=int, default=-1, help="If > 0, limit to this many images. Otherwise, run all.")
    parser.add_argument('--resume', action='store_true', help="Skip images already present in results CSV")
    
    args = parser.parse_args()

    # 1. Model Environment Setup
    me = ModelEnv(args.model)
    
    # FIXED: Check multiple possible checkpoint locations FIRST
    checkpoint_paths = [
        f"/kaggle/working/sloc/models/{args.model}_rsna_best.pth",
        f"/kaggle/working/models/{args.model}_rsna_best.pth",
        f"/kaggle/working/{args.model}_rsna_best.pth"
    ]
    
    checkpoint_loaded = False
    checkpoint = None
    checkpoint_path = None
    
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            checkpoint = torch.load(path, map_location=me.device)
            checkpoint_loaded = True
            break
    
    if checkpoint_loaded:
        print(f"Loading custom RSNA weights from {checkpoint_path}")
        
        # Debug: Print checkpoint keys to understand structure
        head_keys = [k for k in checkpoint.keys() if 'head' in k or (k.startswith('fc.') and not 'mlp' in k)]
        print(f"DEBUG: Head/FC keys in checkpoint: {head_keys[:5]}..." if len(head_keys) > 5 else f"DEBUG: Head/FC keys: {head_keys}")
        
        # Detect checkpoint structure
        has_head_fc = any('head.fc.' in k for k in checkpoint.keys())
        
        if has_head_fc:
            # Checkpoint has nested head.fc structure (Swin models)
            print("DEBUG: Detected head.fc structure (Swin model)")
            if hasattr(me.model, 'head'):
                in_features = me.model.head.in_features
                # Create a module that matches the checkpoint structure
                class HeadWithFC(torch.nn.Module):
                    def __init__(self, in_features, out_features):
                        super().__init__()
                        self.fc = torch.nn.Linear(in_features, out_features)
                    def forward(self, x):
                        return self.fc(x)
                
                me.model.head = HeadWithFC(in_features, 241).to(me.device)
                print(f"DEBUG: Created HeadWithFC structure with {in_features} -> 241")
        else:
            # Standard structure (ResNet or simple ViT)
            print("DEBUG: Detected standard head/fc structure")
            if hasattr(me.model, 'head'):
                me.model.head = torch.nn.Linear(me.model.head.in_features, 241)
            elif hasattr(me.model, 'fc'):
                me.model.fc = torch.nn.Linear(me.model.fc.in_features, 241)
        
        # Move model to device BEFORE loading checkpoint
        me.model.to(me.device)
        
        # Now load the checkpoint
        missing_keys, unexpected_keys = me.model.load_state_dict(checkpoint, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {unexpected_keys}")
        
        # Verify the model output shape
        test_input = torch.randn(1, 3, 224, 224).to(me.device)
        with torch.no_grad():
            test_output = me.model(test_input)
        print(f"DEBUG: Raw model output shape: {test_output.shape}")
        
    else:
        print(f"CRITICAL: {args.model} checkpoint NOT found. Explaining pre-trained features instead.")
        # Modify head to 241 classes for pretrained models
        if hasattr(me.model, 'head'):
            me.model.head = torch.nn.Linear(me.model.head.in_features, 241)
        elif hasattr(me.model, 'fc'):
            me.model.fc = torch.nn.Linear(me.model.fc.in_features, 241)
        me.model.to(me.device)
    
    # Set to eval mode
    me.model.eval()
    
    # Create normalized model wrapper for all operations
    model_wrapper = ModelOutputNormalizer(me.model, me.device)
    
    # 2. Paper Calibration
    modern_archs = ['vit', 'swin', 'convnext', 'dual']
    p = 0.3 if any(arch in args.model.lower() for arch in modern_archs) else 0.6
    config = {'segsize': [16, 32, 48], 'nmasks': [800, 600, 400], 'pprob': [p]*3}

    # 3. Creator Selection - Pass model wrapper to SLOC_m
    if args.variant == 'SLOC_m': 
        creator = SlocM_Creator(model_wrapper=model_wrapper, **config)
    elif args.variant == 'SLOC_xp': 
        creator = SlocExplanationCreator(**config)
    else: 
        creator = AutoProbSlocExplanationCreator(**config)

    # 4. Dataset Loading
    if args.dataset == 'rsna_boneage':
        BONEAGE_ROOT = "/kaggle/working/rsnadata/RSNA_original14236_images"
        from dataset import RSNABoneAgeSource, load_boneage_as_pil
        
        source = RSNABoneAgeSource(BONEAGE_ROOT, split=args.split)
        all_images = list(source.get_all_images().values())
        
        if args.num_images > 0 and args.num_images < len(all_images):
            random.seed(1234)
            selected_images = random.sample(all_images, args.num_images)
            print(f"Running subset: {len(selected_images)} images from {args.split} split.")
        else:
            selected_images = all_images
            print(f"Running FULL {args.split} split: {len(selected_images)} images.")
    else:
        VOC_ROOT = "/kaggle/input/pascalvoc/VOCdevkit/VOC2012"
        selected_images = get_voc_val_images(VOC_ROOT) 
    
    # 5. Output Setup
    output_csv = f"sloc_{args.variant.lower()}_{args.model}_{args.dataset}_{args.split}_results.csv"
    
    processed_images = set()
    if args.resume and os.path.exists(output_csv):
        try:
            existing_df = pd.read_csv(output_csv)
            raw_list = existing_df['Image'].tolist()
            processed_images = set([str(x) for x in raw_list] + [int(x) for x in raw_list if str(x).isdigit()])
            print(f"Resuming: Skipping {len(raw_list)} already processed images.")
        except Exception as e: 
            print(f"Starting fresh: CSV read error: {e}")

    # 6. Benchmark Loop
    print(f"Benchmark Configuration: {args.variant} | {args.model} | {args.split}")
    
    visuals_dir = f"visuals_{args.model}_{args.split}"
    os.makedirs(visuals_dir, exist_ok=True)
    
    start_time = time.time()

    for i, info in enumerate(selected_images):
        image_name = info.name
        if image_name in processed_images: 
            continue

        try:
            if args.dataset == 'rsna_boneage':
                img_pil = load_boneage_as_pil(info.path)
                inp = me.get_transform()(img_pil).unsqueeze(0).to(me.device)
            else:
                img_pil, inp = me.get_image_ext(info.path)
            
            # Use wrapped model for prediction to get consistent output
            logits = model_wrapper(inp)
            target = torch.argmax(logits).item()
            
            sal_numpy = creator.explain(me, inp, target, image_id=image_name)
            visual_path = os.path.join(visuals_dir, f"{image_name}.png")
            save_visual_result(img_pil, sal_numpy, visual_path)
            
            # Use original model (wrapped internally) for evaluation
            m = SLOCPaperEvaluator(me.model, me.device).run(inp, sal_numpy, target, steps=50)
            
            m['Image'] = image_name
            m['GroundTruth_Age'] = info.target
            m['Sex'] = info.desc
            m['Prediction'] = target
            m['Split'] = args.split

            print(f"[{i+1}/{len(selected_images)}] {image_name} (Age: {info.target}m)")
            print(f"   IDD ↑: {m['IDD']:.4f} | NPD ↑: {m['NPD']:.4f} | AIC ↑: {m['AIC']:.4f} | SIC ↑: {m['SIC']:.4f}")
            print(f"   DEL ↓: {m['DEL']:.4f} | INS ↑: {m['INS']:.4f} | POS ↓: {m['POS']:.4f} | NEG ↑: {m['NEG']:.4f}")

            pd.DataFrame([m]).to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)

        except Exception as e:
            print(f"FAILED for {image_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nBenchmark Complete. Total Time: {(time.time()-start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()