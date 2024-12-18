import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
from typing import Callable
import pandas as pd
import dask.dataframe as dd
from dask.delayed import delayed
import shutil
import numpy as np

from src.config import Config
from src.utils import get_logger, plot_class_distribution
from src.augmentation import get_base_transforms, get_augmentation_list, get_resize_transform

logger = get_logger(__name__)


class MushroomDataset(Dataset):
    """Base dataset for loading mushroom images from disk using Dask for lazy loading."""
    
    def __init__(self, root_dir: str | Path, transform: Callable = None, skip_transform: bool = False):
        """
        Args:
            root_dir (str or Path): Directory containing mushroom species folders
            transform (callable, optional): Transform to be applied to images
            skip_transform (bool): If True, returns PIL images without transforms
        """
        self.root_dir = Path(root_dir)
        self.base_transform = get_resize_transform()  # Always resize
        self.train_transform = transform  # Additional transforms for training (normalization)
        self.skip_transform = skip_transform
        
        random.seed(Config.SEED)
        torch.manual_seed(Config.SEED)
        
        self._setup_dataset()
        self._print_statistics()
        
    def _setup_dataset(self):
        """Initialize dataset structure and load image paths using Dask."""
        # list of all image paths and their species
        image_data = []
        for ext in ['.jpg', '.jpeg', '.png']:
            for img_path in self.root_dir.rglob(f'*{ext}'):
                species_name = img_path.parent.name
                image_data.append({'path': str(img_path), 'species': species_name})

        # Dask DataFrame
        self.ddf = dd.from_pandas(pd.DataFrame(image_data), npartitions=4)
        
        # Create class mappings
        self.classes = sorted(self.ddf['species'].unique().compute())
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        Config.NUM_CLASSES = len(self.classes) # set number of classes
        
        # Convert species to indices and compute the final DataFrame
        self.df = self.ddf.assign(
            label=self.ddf['species'].map(self.class_to_idx, meta=('label', 'int32'))
        ).compute()
        
        # explicit lists for images and labels for easy indexing
        self.images = self.df['path'].tolist()
        self.labels = self.df['label'].tolist()
        
        # Create species paths mapping for statistics
        self.species_paths = defaultdict(list)
        for path, species in zip(self.images, self.df['species']):
            self.species_paths[species].append(path)

    def _load_image(self, path: str) -> Image.Image:
        """Load an image using Dask for lazy loading."""
        @delayed
        def read_image(img_path):
            return Image.open(img_path).convert('RGB')
        
        try:
            # Create delayed object for lazy loading, which is only computed when needed
            delayed_img = read_image(path)
            return delayed_img.compute()
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            raise e

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
            image = self._load_image(img_path)
            image = self.base_transform(image)
            if not self.skip_transform and self.train_transform:
                image = self.train_transform(image)

        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        label = self.labels[idx]
        return image, label

class BalancedDataset(Dataset):
    """Dataset wrapper that balances class distribution through on-the-fly augmentation."""
    # needed to switch to on-the-fly augmentation due to OOM errors. if this is not sufficient, we would need to use e.g. Dask to parallelize the augmentation.
    
    def __init__(self, original_dataset: Dataset, augmentation_list: list[Callable], max_aug_factor: int = None):
        """
        Args:
            original_dataset: Base dataset to balance
            augmentation_list: List of augmentations to apply
            max_aug_factor: Maximum number of augmentations per image
        """
        base_dataset = original_dataset.dataset
        base_dataset.skip_transform = True
        
        self.original_dataset = original_dataset
        self.augmentation_list = augmentation_list
        self.transform = base_dataset.transform
        
        # Calculate augmentation factors per class
        labels = [label for _, label in self.original_dataset]
        class_counts = torch.bincount(torch.tensor(labels))
        max_class_count = class_counts.max().item()
        
        self.sample_indices = []
        self.aug_factors = []
        
        for idx, label in enumerate(labels):
            original_count = class_counts[label].item()
            aug_factor = min(
                max_aug_factor,
                (max_class_count + original_count - 1) // original_count
            )
            
            # Add original and augmented versions
            self.sample_indices.extend([idx] * aug_factor)
            self.aug_factors.extend(range(aug_factor))  # 0 means original, >0 means augmented
        
        base_dataset.skip_transform = False # reset skip_transform
        
        self._print_distribution()
    
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
        
        label_counts = defaultdict(int)
        for idx in self.sample_indices:
            label = self.original_dataset[idx][1]
            label_counts[label] += 1
        
        for class_idx in range(len(class_names)):
            species_name = class_names[class_idx]
            count = label_counts[class_idx]
            logger.info(f"{species_name:<{max_name_length}} | {count:>6} images")

    def __len__(self):
        return len(self.sample_indices)

    # need to use Dask to parallelize this?
    def __getitem__(self, idx):
        """Get item with on-the-fly augmentation."""
        original_idx = self.sample_indices[idx]
        aug_factor = self.aug_factors[idx]
        
        # Get PIL image without transforms
        pil_image, label = self.original_dataset[original_idx]
        
        # Apply augmentation if needed (augmentations return PIL images)
        if aug_factor > 0:
            pil_image = random.choice(self.augmentation_list)(pil_image)
        
        # Apply base transforms (converts to tensor)
        if self.transform:
            image = self.transform(pil_image)
        else:
            image = pil_image
        
        return image, label


def weighted_train_test_split(indices, labels, weights, test_size, random_state=None):
    """
    Custom splitting function that respects both weights and stratification.
    
    Args:
        indices: List of indices to split
        labels: List of labels corresponding to indices
        weights: Array of weights for each sample
        test_size: Proportion of samples to include in test set
        random_state: Random seed for reproducibility
    
    Returns:
        train_indices, test_indices
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.array(indices)
    labels = np.array(labels)
    weights = np.array(weights)
    
    unique_labels = np.unique(labels)
    train_indices = []
    test_indices = []
    
    for label in unique_labels:
        # Get indices for this class
        class_mask = labels == label
        class_indices = indices[class_mask]
        class_weights = weights[class_mask]
        
        # normalize weights to sum to 1
        class_weights = class_weights / class_weights.sum()
        
        n_samples = len(class_indices)
        n_test = int(np.round(test_size * n_samples))
        
        # Random weighted sampling without replacement
        test_idx = np.random.choice(
            class_indices, 
            size=n_test, 
            replace=False, 
            p=class_weights
        )
        train_idx = np.setdiff1d(class_indices, test_idx)
        
        train_indices.extend(train_idx)
        test_indices.extend(test_idx)
    
    return train_indices, test_indices

def create_static_splits(data_dir: Path, output_dir: Path, train_val_split=0.2, val_test_split=0.5) -> tuple[Path, Path, Path]:
    """
    1. Load original dataset
    2. Calculate target distribution after augmentation
    3. Create weighted splits
    4. Save splits to separate directories for later use
    """
    # Load original dataset
    original_dataset = MushroomDataset(
        root_dir=data_dir,
        transform=get_base_transforms()
    )

    # Calculate augmentation factors per class
    class_counts = defaultdict(int)
    for _, label in original_dataset:
        class_counts[label] += 1

    plot_class_distribution(
        data=class_counts,
        title="Pre-augmentation class distribution",
        xlabel="Class",
        ylabel="Number of images",
        save_path=Config.PLOTS_DIR / "pre_augmentation_class_distribution.png"
    )

    max_class_count = max(class_counts.values())
    target_counts = {}
    for label, count in class_counts.items():
        # Calculate how many augmentations we need to reach max_class_count
        needed_total = max_class_count
        possible_total = count * Config.MAX_AUGMENTATION_FACTOR_PER_IMAGE
        target_counts[label] = min(needed_total, possible_total)
    
    plot_class_distribution(
        data=target_counts,
        title="Post-augmentation class distribution",
        xlabel="Class",
        ylabel="Number of images",
        save_path=Config.PLOTS_DIR / "post_augmentation_class_distribution.png"
    )


    # Calculate split weights based on target distribution
    weights = np.array([target_counts[label] for _, label in original_dataset])
    weights = weights / weights.sum()
    
    # Create stratified splits with weighted sampling
    indices = list(range(len(original_dataset)))
    labels = [label for _, label in original_dataset]
    
    # First split: train vs val+test
    train_indices, valtest_indices = weighted_train_test_split(
        indices=indices,
        labels=labels,
        weights=weights,
        test_size=train_val_split,
        random_state=Config.SEED
    )
    
    # Second split: val vs test
    valtest_labels = [labels[i] for i in valtest_indices]
    valtest_weights = weights[valtest_indices]
    
    val_indices, test_indices = weighted_train_test_split(
        indices=valtest_indices,
        labels=valtest_labels,
        weights=valtest_weights,
        test_size=val_test_split,
        random_state=Config.SEED
    )
    
    splits = {
        'train': (train_indices, True),  # (indices, should_augment)
        'val': (val_indices, False),
        'test': (test_indices, False)
    }
    
    logger.info(f"Creating static splits with augmentation factor {Config.MAX_AUGMENTATION_FACTOR_PER_IMAGE}")
    logger.info(f"Target counts: {target_counts}")

    for split_name, (split_indices, should_augment) in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in split_indices:
            img_path = original_dataset.images[idx]
            species = original_dataset.df.iloc[idx]['species']
            species_dir = split_dir / species
            species_dir.mkdir(exist_ok=True)
            
            shutil.copy2(img_path, species_dir)
            
            if should_augment:
                image = Image.open(img_path).convert('RGB')
                base_name = Path(img_path).stem
                extension = Path(img_path).suffix
                
                current_label = labels[idx]
                target_count = target_counts[current_label]
                original_count = class_counts[current_label]
                
                # Calculate augmentations needed for this specific image
                augs_per_image = min(
                    Config.MAX_AUGMENTATION_FACTOR_PER_IMAGE - 1,  # -1 because we already have original
                    (target_count - original_count + idx) // original_count  # Distribute remaining needed augs
                )
                
                for aug_idx in range(augs_per_image):
                    aug = random.choice(get_augmentation_list())
                    aug_image = aug(image)
                    aug_path = species_dir / f"{base_name}_aug{aug_idx + 1}{extension}"
                    aug_image.save(aug_path)
    
    logger.info("Static splits created successfully!")

    return output_dir / 'train', output_dir / 'val', output_dir / 'test'

def load_prepared_datasets(train_dir: Path, val_dir: Path, test_dir: Path) -> tuple[MushroomDataset, MushroomDataset, MushroomDataset]:
    """Load pre-split datasets."""
    train_dataset = MushroomDataset(train_dir, transform=get_base_transforms()) # this should already have augmentation
    val_dataset = MushroomDataset(val_dir, transform=get_base_transforms()) # this should not have augmentation
    test_dataset = MushroomDataset(test_dir, transform=get_base_transforms()) # this should not have augmentation
    
    logger.info(f"Training set: {len(train_dataset)} images")
    logger.info(f"Validation set: {len(val_dataset)} images")
    logger.info(f"Test set: {len(test_dataset)} images")
    
    return train_dataset, val_dataset, test_dataset