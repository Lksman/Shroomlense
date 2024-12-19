from torchvision import transforms
from src.config import Config
from src.utils import get_logger

logger = get_logger(__name__)

def get_base_transforms():
    """Basic transforms applied to all images (resize + convert to tensor)"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),  # converts to [0,1] range
    ])

def get_training_transforms(mean, std):
    """Combined transforms for model training"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),  # Converts to [0,1]
        transforms.Normalize(mean=mean, std=std)  # Normalizes using dataset statistics
    ])

def get_augmentation_list(aug_rotation:bool=True, aug_color_jitter:bool=True, aug_affine:bool=False):
    """Returns a list of augmentation-only transforms"""
    augmentation_list = []
    if aug_rotation:    
        augmentation_list.append(
            # Augmentation 1: Horizontal flip + slight rotation + zoom
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5), # flip horizontally 50% of the time
                transforms.RandomRotation(30),
                # Zoom in 15% to alleviate black regions caused by rotation.
                # The largest inscribed square after a 30-degree rotation has a side length ~cos(30) ≈ 0.866 of the original.
                # We use 0.85 (slightly less) to ensure all black borders are cropped.
                transforms.CenterCrop(int(Config.IMAGE_SIZE * 0.85)),  
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))  # Ensure consistent size
            ])
        )
    if aug_color_jitter:
        augmentation_list.append(
            # Augmentation 2: Color jittering
            transforms.Compose([
                transforms.ColorJitter(
                    brightness=0.2, 
                    contrast=0.2, 
                    saturation=0.2, 
                ),
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))  # Ensure consistent size
            ])
        )
    if aug_affine:
        augmentation_list.append(
            # Augmentation 3: Random affine transformation
            transforms.Compose([
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1)
                ),
                transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))  # Ensure consistent size
            ])
        )
    return augmentation_list