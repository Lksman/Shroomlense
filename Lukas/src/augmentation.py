from torchvision import transforms
from src.config import Config
import os
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import random
from src.utils import get_logger

logger = get_logger(__name__)

def get_base_transforms():
    """Basic transforms applied to all images during training"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=Config.DATASET_MEAN,
            std=Config.DATASET_STD
        )
    ])

def get_resize_transform():
    """Only resize transform for saving augmented images"""
    return transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE))
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

def create_augmented_dataset(root_dir: str | Path, output_dir: str | Path = None):
    """
    Create and save augmented versions of all images in the dataset.
    
    Args:
        root_dir: Root directory containing the class folders
        output_dir: Directory to save augmented images (defaults to root_dir if None)
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir) if output_dir else root_dir
    
    # Get augmentation transforms (without normalization)
    augmentation_list = get_augmentation_list(
        aug_rotation=True,
        aug_color_jitter=True,
        aug_affine=False
    )
    
    # Process all images
    for category in ['deadly', 'edible', 'not_edible', 'poisonous']:
        category_path = root_dir / category
        if not category_path.exists():
            continue
            
        logger.info(f"Processing {category} category...")
        
        # Process each species folder
        for species_dir in category_path.iterdir():
            if not species_dir.is_dir():
                continue
                
            species_name = species_dir.name
            logger.info(f"Processing species: {species_name}")
            
            # Create output directory
            output_species_dir = output_dir / category / species_name
            output_species_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all images in the species directory
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files.extend(species_dir.glob(f'*{ext}'))
            
            if not image_files:
                continue
                
            # Calculate augmentation factor
            max_aug_factor = min(
                Config.MAX_AUGMENTATION_FACTOR_PER_IMAGE,
                Config.MAX_IMAGES_PER_CLASS // len(image_files)
            )
            
            # Process each image
            for img_path in tqdm(image_files, desc=f"Augmenting {species_name}"):
                try:
                    # Load original image
                    image = Image.open(img_path).convert('RGB')
                    base_name = img_path.stem
                    extension = img_path.suffix
                    
                    # Save original image if output_dir is different
                    if output_dir != root_dir:
                        image.save(output_species_dir / f"{base_name}{extension}")
                    
                    # Create augmented versions
                    for aug_idx in range(max_aug_factor - 1):  # -1 because we already have the original
                        # Randomly choose an augmentation
                        augmentation = random.choice(augmentation_list)
                        
                        # Apply augmentation
                        aug_image = augmentation(image)
                        
                        # Save augmented image
                        aug_name = f"{base_name}_aug{aug_idx + 1}{extension}"
                        aug_image.save(output_species_dir / aug_name)
                        
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue
                    
    logger.info("Augmentation complete!")
