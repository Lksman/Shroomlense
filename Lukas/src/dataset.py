import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Callable

from src.config import Config
from src.utils import get_logger
from src.augmentation import get_base_transforms, get_augmentation_list

logger = get_logger(__name__)


class MushroomDataset(Dataset):
    """Base dataset for loading mushroom images from disk."""
    
    def __init__(self, root_dir: str | Path, transform: Callable = None, skip_transform: bool = False):
        """
        Args:
            root_dir (str or Path): Directory containing mushroom species folders
            transform (callable, optional): Transform to be applied to images
            skip_transform (bool): If True, returns PIL images without transforms
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.skip_transform = skip_transform
        
        random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        
        self._setup_dataset()
        self._print_statistics()
        
    def _setup_dataset(self):
        """Initialize dataset structure and load image paths."""
        self.species_paths = defaultdict(list)
        image_extensions = ['.jpg', '.jpeg', '.png']
        for ext in image_extensions:
            for img_path in self.root_dir.rglob(f'*{ext}'):
                species_name = img_path.parent.name
                self.species_paths[species_name].append(img_path)
            
        # class mappings
        self.classes = sorted(self.species_paths.keys())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        Config.NUM_CLASSES = len(self.classes)
        
        # flatten lists
        self.images = []
        self.labels = []
        for species_name, paths in self.species_paths.items():
            self.images.extend(paths)
            self.labels.extend([self.class_to_idx[species_name]] * len(paths))

    def _print_statistics(self):
        """Print dataset statistics."""
        max_name_length = max(len(name) for name in self.classes)
        logger.info("Dataset Statistics:")
        logger.info(f"Number of species: {len(self.classes)}")
        logger.info(f"Total images: {len(self.images)}")
        logger.info("Class distribution:")
        
        logger.info(f"{'Species':<{max_name_length}} | {'Count':>6} images")
        logger.info("-" * (max_name_length + 15))
    
        for species_name in self.classes:
            count = len(self.species_paths[species_name])
            logger.info(f"{species_name:<{max_name_length}} | {count:>6} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, label) where label is the class index
        """
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
            
        label = self.labels[idx]
        
        if self.transform and not self.skip_transform:
            image = self.transform(image)
        
        return image, label

class BalancedDataset(Dataset):
    """Dataset wrapper that balances class distribution through augmentation."""
    
    def __init__(self, original_dataset: Dataset, augmentation_list: list[Callable], max_aug_factor: int = None):
        """
        Args:
            original_dataset: Base dataset to balance
            augmentation_list: List of augmentations to apply
            max_aug_factor: Maximum number of augmentations per image
        """
        # Get base dataset from Subset and set it to skip transforms
        base_dataset = original_dataset.dataset
        base_dataset.skip_transform = True
        
        self.original_dataset = original_dataset
        self.augmentation_list = augmentation_list
        self.transform = base_dataset.transform  # Store transform for later use
        
        self.augmented_images = []
        self.augmented_labels = []
        self._balance_dataset(max_aug_factor)
        self._print_distribution()
        
        # Reset skip_transform
        base_dataset.skip_transform = False

    def _balance_dataset(self, max_aug_factor: int):
        """Balance dataset through augmentation.
        
        Args:
            max_aug_factor: Maximum number of augmented versions per original image
        """
        logger.info(f"Balancing dataset with max augmentation factor {max_aug_factor}, this might take a while...")

        labels = [label for _, label in self.original_dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        max_class_count = class_counts.max().item()
        
        for class_idx in tqdm(range(len(class_counts)), desc="Augmenting classes", leave=False):
            class_indices = [i for i, label in enumerate(labels) if label == class_idx]
            original_count = len(class_indices)
            
            target_count = max_class_count
            remaining_images = target_count - original_count
            
            if remaining_images <= 0:
                # just add originals
                for idx in class_indices:
                    pil_image, label = self.original_dataset[idx]
                    tensor_image = self.transform(pil_image)
                    self.augmented_images.append(tensor_image)
                    self.augmented_labels.append(label)
                continue
            
            aug_per_image = min(
                max_aug_factor,  # Don't exceed max augmentations per image
                (remaining_images + original_count - 1) // original_count  # ceil
            )
            
            for idx in class_indices:
                pil_image, label = self.original_dataset[idx]
                
                # Always add original image
                tensor_image = self.transform(pil_image)
                self.augmented_images.append(tensor_image)
                self.augmented_labels.append(label)
                
                # Add augmented versions
                for _ in range(aug_per_image):
                    aug = random.choice(self.augmentation_list)
                    aug_image = aug(pil_image)
                    tensor_image = self.transform(aug_image)
                    self.augmented_images.append(tensor_image)
                    self.augmented_labels.append(label)

    def _print_distribution(self):
        """Print class distribution after balancing."""
        base_dataset = self.original_dataset.dataset
        class_names = base_dataset.classes
        max_name_length = max(len(name) for name in class_names)
        
        logger.info("")
        logger.info("Balanced Dataset Statistics:")
        logger.info(f"Total images: {len(self)}")
        logger.info("Class distribution after balancing:")
        
        logger.info(f"{'Species':<{max_name_length}} | {'Count':>6} images")
        logger.info("-" * (max_name_length + 15))
        
        new_counts = torch.bincount(torch.tensor(self.augmented_labels))
        for class_idx, count in enumerate(new_counts):
            species_name = class_names[class_idx]
            logger.info(f"{species_name:<{max_name_length}} | {count:>6} images")

    def __len__(self):
        return len(self.augmented_images)

    def __getitem__(self, idx):
        return self.augmented_images[idx], self.augmented_labels[idx]



def prepare_datasets(data_dir, val_split=0.2, augment: bool =True):
    """Prepare and split datasets with stratified sampling"""

    logger.info(f"Loading base dataset from {data_dir}")
    base_dataset = MushroomDataset(
        root_dir=data_dir,
        transform=get_base_transforms()
    )

    indices = list(range(len(base_dataset)))
    labels = [label for _, label in base_dataset]
    
    # stratified split. TODO: take class-level augmentations into account
    logger.info(f"Splitting dataset into {(1 - val_split) * 100}% training and {val_split * 100}% validation set (stratified).")
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_split,
        stratify=labels,
        random_state=Config.SEED
    )
    
    logger.info(f"Creating training subset with {len(train_idx)} images.")
    train_subset = torch.utils.data.Subset(base_dataset, train_idx)
    
    logger.info(f"Creating validation subset with {len(val_idx)} images")
    val_subset = torch.utils.data.Subset(base_dataset, val_idx)
    
    if augment:
        train_dataset = BalancedDataset(
            original_dataset=train_subset,
            augmentation_list=get_augmentation_list(Config.AUGMENTATION_STEPS),
            max_aug_factor=Config.MAX_AUGMENTATION_FACTOR_PER_CLASS
        )
    else:
        train_dataset = train_subset

    return train_dataset, val_subset