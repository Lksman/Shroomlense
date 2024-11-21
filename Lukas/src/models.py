from torch.types import Tensor
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class MushroomVGG19(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # Load pretrained VGG19
        self.vgg19 = vgg19(weights=VGG19_Weights.DEFAULT)
        
        # Replace the classifier
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.vgg19(x)

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