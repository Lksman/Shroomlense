from torchvision import transforms
from src.config import Config

def get_base_transforms():
    """Basic transforms applied to all images"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=Config.MEAN,
            std=Config.STD
        )
    ])

def get_augmentation_list(aug_rotation:bool=True, aug_color_jitter:bool=True, aug_affine:bool=False):
    """Returns a list of augmentation-only transforms"""
    augmentation_list = []
    if aug_rotation:    
        augmentation_list.append(
            # Augmentation 1: Horizontal flip + slight rotation
            transforms.Compose([
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.RandomRotation(15),
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
                )
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
                )
            ])
        )
    return augmentation_list
