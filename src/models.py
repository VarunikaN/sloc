import os
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image

import timm, torchray
import torchray.benchmark


class ModelEnv:
    def __init__(self, arch, resolution=224, weights_path=None, regression=True):
        self.arch = arch
        self.device = self.get_device()
        self.shape = (resolution, resolution)
        self.regression = regression
        self.model = self.load_model(self.arch, self.device, weights_path)

    def load_model(self, arch, dev, weights_path=None):
        import os
        # Load the base architecture
        if self.regression:
        # Replace the classification layer with a single output for age
            if 'vit' in arch or 'swin' in arch or 'convnext' in arch:
                model = timm.create_model(arch, pretrained=True, num_classes=1)
            else:
                model = torchvision.models.__dict__[arch](weights='DEFAULT')
                if hasattr(model, 'fc'):
                    model.fc = nn.Linear(model.fc.in_features, 1)
                elif hasattr(model, 'classifier'):
                    # Handle VGG/DenseNet style classifiers
                    if isinstance(model.classifier, nn.Sequential):
                        in_features = model.classifier[-1].in_features
                        model.classifier[-1] = nn.Linear(in_features, 1)
                    else:
                        model.classifier = nn.Linear(model.classifier.in_features, 1)

        model = timm.create_model(arch, pretrained=True, num_classes=241 if not self.regression else 1)
    
        if weights_path and os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}")
            model.load_state_dict(torch.load(weights_path, map_location=dev))
            
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
            return self.model.conv_head
        elif 'swin' in a:
            return self.model.layers[-1].blocks[-1]
        elif 'regnet' in a:
            # For torchvision regnet
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
        # Add 'mobilenet', 'efficientnet', and 'regnet' to this condition
        elif any(x in self.arch for x in ['resnet', 'vgg', 'convnext', 'densenet', 'mobilenet', 'efficientnet', 'regnet']):
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.shape),
                torchvision.transforms.CenterCrop(self.shape),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225]),
            ])
        elif 'vit' in self.arch or 'swin' in self.arch:
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
        # Pre-process the image and convert into a tensor
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
        self.loss  = nn.CrossEntropyLoss()

    def forward(self, x):
        if type(self.k) == int:            
            values = torch.stack([x], dim=-1)
            result = values[:,self.k,:]            
        else:
            result = x[:, self.k]                            
        
        return result
