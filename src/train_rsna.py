import os, sys, torch, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add your local src path
sys.path.insert(0, "/home/iiitdmk-drnagaraju/xai/sloc/src")
from models import ModelEnv
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
    parser.add_argument('--model', type=str, default='resnet50', help="Model name from models.py or timm")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    # Load Model Env & Transform
    me = ModelEnv(args.model)
    device = me.device
    model = me.model
    
    # Adapt head for RSNA (Classification of ages 1-240 months)
    if hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, 241)
    elif hasattr(model, 'head'):
        model.head = nn.Linear(model.head.in_features, 241)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Dataset Setup
    root = "/home/iiitdmk-drnagaraju/aps/RSNA_original14236_images"
    train_ds = RSNAHandDataset(root, 'training', me.get_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Training Loop
    print(f"Starting finetuning for {args.model}...")
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    os.makedirs("models", exist_ok=True)
    save_path = f"models/{args.model}_rsna_best.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()