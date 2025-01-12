import torch
from torch.types import Tensor
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights
from transformers import ViTForImageClassification
from src.config import Config
from src.utils import set_all_seeds

class MushroomVGG19(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.vgg19(x)


class MushroomViT(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
        self.vit.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # expects shape: (batch_size, channels, height, width)
        outputs = self.vit(x)
        return outputs.logits
    

class MushroomCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            # conv block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            
            # conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x 


def create_model(model_name: str, num_classes: int) -> tuple[torch.nn.Module, dict]:
    """Create model and get its config."""
    set_all_seeds()
    
    model_mapping = {
        'cnn': MushroomCNN,
        'vgg19': MushroomVGG19,
        'vit': MushroomViT
    }
    
    if model_name not in Config.MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = Config.MODEL_CONFIGS[model_name]
    model_class = model_mapping[model_name]
    model = model_class(num_classes)
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    return model, config