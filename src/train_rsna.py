import os, sys, torch, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import timm

# Add your local src path
sys.path.insert(0, "/kaggle/working/sloc/src")
from dataset import RSNABoneAgeSource, load_boneage_as_pil

class RSNAHandDataset(Dataset):
    def __init__(self, root, split, transform):
        self.source = RSNABoneAgeSource(root, split=split)
        self.images = list(self.source.get_all_images().values())
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        info = self.images[idx]
        img = load_boneage_as_pil(info.path)
        return self.transform(img), info.target

def main():
    parser = argparse.ArgumentParser(description="Finetune any model on RSNA Bone Age.")
    parser.add_argument('--model', type=str, default='resnet50', help="Model name from timm")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # --- UNIVERSAL BACKBONE INITIALIZATION ---
    print(f"Initializing {args.model}...")
    # Use timm to automatically handle head replacement and pooling for ALL backbones
    model = timm.create_model(
        args.model, 
        pretrained=True, 
        num_classes=241, 
        global_pool='avg' # Ensures 3D feature maps are pooled into 1D vectors
    )
    
    # Get model-specific transforms (crucial for ViT/Swin/ConvNeXt)
    data_config = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_config, is_training=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) # AdamW is better for ViTs
    
    # Dataset Setup
    root = "/kaggle/working/rsnadata/RSNA_original14236_images"
    train_ds = RSNAHandDataset(root, 'training', transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            
            # Final Safeguard: Flatten any residual spatial dims to prevent the 3D target error
            if outputs.dim() > 2:
                outputs = torch.mean(outputs, dim=tuple(range(2, outputs.dim())))
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{args.model}_rsna_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()