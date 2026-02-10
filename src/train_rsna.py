import os, sys, torch, argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not available. Only torchvision models will work.")

# Add your local src path
sys.path.insert(0, "/kaggle/working/sloc/src")
from dataset import RSNABoneAgeSource, load_boneage_as_pil


class RSNAHandDataset(Dataset):
    def __init__(self, root, split, transform):
        self.source = RSNABoneAgeSource(root, split=split)
        self.images = list(self.source.get_all_images().values())
        self.transform = transform
    
    def __len__(self): 
        return len(self.images)
    
    def __getitem__(self, idx):
        info = self.images[idx]
        img = load_boneage_as_pil(info.path)
        return self.transform(img), info.target


def load_model_universal(model_name, num_classes=241):
    """Universal model loader that correctly handles torchvision and timm."""
    
    # Check if it's a torchvision model
    is_torchvision = model_name in torchvision.models.__dict__
    
    # Timm-exclusive models (not in torchvision)
    timm_only = ['deit', 'beit', 'vit_tiny', 'vit_small']
    is_timm_only = any(prefix in model_name.lower() for prefix in timm_only)
    
    if is_torchvision:
        # Load from torchvision
        print(f"Loading {model_name} from torchvision...")
        model = torchvision.models.__dict__[model_name](weights='DEFAULT')
        
        # Modify head for 241 classes
        if hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
            else:
                model.head = nn.Linear(model.head.in_features, num_classes)
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        # Get default torchvision transform
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]),
        ])
        
    elif TIMM_AVAILABLE and (is_timm_only or model_name in timm.list_models()):
        # Load from timm
        print(f"Loading {model_name} from timm...")
        
        # DeiT distilled models need special handling
        global_pool_mode = 'token' if 'distilled' in model_name else 'avg'
        
        try:
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=num_classes,
                global_pool=global_pool_mode
            )
        except TypeError:
            # Fallback for models that don't support global_pool parameter
            model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        
        # Get timm's recommended transform for this model
        data_config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**data_config, is_training=True)
        
    else:
        available_tv = list(torchvision.models.__dict__.keys())[:10]
        available_timm = timm.list_models()[:10] if TIMM_AVAILABLE else []
        raise RuntimeError(
            f"Model '{model_name}' not found!\n"
            f"Torchvision models (sample): {available_tv}\n"
            f"Timm models (sample): {available_timm}\n"
            f"Install timm if needed: pip install timm"
        )
    
    return model, transform


class OutputNormalizer(nn.Module):
    """Ensures model output is always [B, C] regardless of architecture."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._is_spatial = None
    
    def forward(self, x):
        out = self.model(x)
        
        # Auto-detect on first call
        if self._is_spatial is None:
            self._is_spatial = len(out.shape) > 2
            if self._is_spatial:
                print(f"Detected spatial output {out.shape}, will pool to [B, C]")
        
        # Pool spatial dimensions if needed
        if self._is_spatial:
            out = out.mean(dim=tuple(range(2, out.dim())))
        
        return out


def main():
    parser = argparse.ArgumentParser(description="Finetune any model on RSNA Bone Age.")
    parser.add_argument('--model', type=str, default='resnet50', 
                       help="Model name from torchvision or timm (e.g., efficientnet_v2_s, deit_tiny_distilled_patch16_224)")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='models', help='Directory to save checkpoints')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"{'='*60}\n")

    # Load model and get appropriate transform
    model, transform = load_model_universal(args.model, num_classes=241)
    
    # Wrap model to handle different output shapes
    model = OutputNormalizer(model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on {device}\n")
    
    # Test forward pass
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = model(test_input)
    print(f"Test output shape: {test_output.shape} (expected: [1, 241])\n")
    
    if test_output.shape != torch.Size([1, 241]):
        raise RuntimeError(f"Output shape mismatch! Got {test_output.shape}, expected [1, 241]")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Dataset
    root = "/kaggle/working/rsnadata/RSNA_original14236_images"
    train_ds = RSNAHandDataset(root, 'training', transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    print(f"Dataset: {len(train_ds)} training images")
    print(f"Starting training...\n")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"{args.model}_rsna_best.pth")
            torch.save(model.model.state_dict(), save_path)  # Save inner model (unwrap OutputNormalizer)
            print(f"✓ Saved best model to {save_path}\n")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
