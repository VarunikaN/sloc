import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    import torchray
    import torchray.benchmark
    TORCHRAY_AVAILABLE = True
except ImportError:
    TORCHRAY_AVAILABLE = False


class ModelEnv:
    def __init__(self, arch, resolution=224, weights_path=None, regression=True):
        self.arch = arch
        self.device = self.get_device()
        self.shape = (resolution, resolution)
        self.regression = regression
        self.model = self.load_model(self.arch, self.device, weights_path)

    def _is_timm_only_model(self, arch):
        """Check if model MUST be loaded from timm (not available in torchvision)."""
        # DeiT is ONLY in timm
        timm_only = ['deit', 'beit', 'vit_tiny', 'vit_small']
        return any(prefix in arch.lower() for prefix in timm_only)
    
    def _is_torchvision_model(self, arch):
        """Check if model is available in torchvision."""
        return arch in torchvision.models.__dict__

    def load_model(self, arch, dev, weights_path=None):
        """Load model from torchvision or timm, prioritizing the correct source."""
        num_classes = 1 if self.regression else 241
        
        # Strategy 1: If it's ONLY in timm (like DeiT), use timm
        if TIMM_AVAILABLE and self._is_timm_only_model(arch):
            print(f"Loading {arch} from timm (timm-exclusive model)...")
            model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
            
        # Strategy 2: If it's in torchvision, prefer torchvision
        elif self._is_torchvision_model(arch):
            print(f"Loading {arch} from torchvision...")
            model = torchvision.models.__dict__[arch](weights='DEFAULT')
            
            # Modify classification head for RSNA (241 classes or 1 for regression)
            if hasattr(model, 'fc'):
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            elif hasattr(model, 'head'):
                if hasattr(model.head, 'fc'):
                    # Nested head.fc (some Swin variants)
                    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
                else:
                    model.head = nn.Linear(model.head.in_features, num_classes)
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
        # Strategy 3: Try timm as fallback for ViT/Swin/ConvNeXt variants
        elif TIMM_AVAILABLE:
            print(f"Loading {arch} from timm (fallback)...")
            try:
                model = timm.create_model(arch, pretrained=True, num_classes=num_classes)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load {arch} from both torchvision and timm.\n"
                    f"Timm error: {e}\n"
                    f"Available in torchvision: {arch in torchvision.models.__dict__}\n"
                    f"Install timm if needed: pip install timm"
                )
        else:
            raise RuntimeError(
                f"Model {arch} not found in torchvision and timm not installed.\n"
                f"Install timm: pip install timm"
            )
        
        # Load custom weights if provided
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=dev)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Warning: Missing keys: {missing[:3]}..." if len(missing) > 3 else f"Warning: Missing keys: {missing}")
            if unexpected:
                print(f"Warning: Unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"Warning: Unexpected keys: {unexpected}")
            
        return model.to(dev).eval()

    def narrow_model(self, catidx, with_softmax=False):
        if self.regression:
            return self.model
        if "voc" in self.arch:
            modules = (
                [self.model] + 
                ([nn.Sigmoid()] if with_softmax else []) +
                [SelectKthLogit(catidx)])
        else:
            modules = (
                [self.model] + 
                ([nn.Softmax(dim=1)] if with_softmax else []) +
                [SelectKthLogit(catidx)])

        return nn.Sequential(*modules)
        
    def get_cam_target_layer(self):
        a = self.arch.lower()
        if 'resnet' in a:
            return self.model.layer4[-1]
        elif 'vgg' in a or 'mobilenet' in a:
            return self.model.features[-1]
        elif 'densenet' in a:
            return self.model.features.norm5
        elif 'convnext' in a:
            return self.model.stages[-1].blocks[-1]
        elif 'efficientnet' in a:
            return self.model.conv_head if hasattr(self.model, 'conv_head') else self.model.features[-1]
        elif 'swin' in a:
            return self.model.layers[-1].blocks[-1]
        elif 'regnet' in a:
            return self.model.trunk_output[-1]
            
        raise Exception(f'Target layer not defined for arch: {self.arch}')
    
    def get_cex_conv_layer(self):
        if self.arch == 'resnet50':
            return self.model.layer4[-1].conv3
        elif self.arch == 'vgg16':
            return self.model.features[-3]
        elif self.arch == 'convnext_base':
            return self.model.stages[-1].blocks[-1].conv_dw
        raise Exception('Unexpected arch')

    def get_device(self, gpu=0):
        device = torch.device(
            f'cuda:{gpu}'
            if torch.cuda.is_available() and gpu is not None
            else 'cpu')
        return device

    def get_transform(self):    
        if "voc" in self.arch:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255.0)
            ])
        elif any(x in self.arch for x in ['resnet', 'vgg', 'convnext', 'densenet', 'mobilenet', 'efficientnet', 'regnet']):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.shape),
                torchvision.transforms.CenterCrop(self.shape),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),
            ])
        elif 'vit' in self.arch or 'swin' in self.arch or 'deit' in self.arch or 'beit' in self.arch:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            assert False, f"unexpected arch: {self.arch}"
        return transform

    def get_image_ext(self, path):
        img = Image.open(path)
        transform = self.get_transform()
        x = transform(img).unsqueeze(0)
        return img, x.to(self.device)

    def get_image(self, path):
        return self.get_image_ext(path)[1]
    

class SelectKthLogit(nn.Module):
    def __init__(self, k):
        super(SelectKthLogit, self).__init__()
        self.k = k        
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        if type(self.k) == int:            
            values = torch.stack([x], dim=-1)
            result = values[:,self.k,:]            
        else:
            result = x[:, self.k]                            
        
        return result
