import shutil
import random
from typing import Callable
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import dask.dataframe as dd
from dask.delayed import delayed
from tqdm import tqdm

from src.config import Config
from src.utils import get_logger, plot_class_distribution
from src.augmentation import get_base_transforms, get_augmentation_list, get_training_transforms

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
        self.base_transform = get_base_transforms()  # Always resize and convert to tensor
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
            # Lazy load image as PIL
            image = self._load_image(img_path)
            
            if not self.skip_transform:
                if self.train_transform:
                    # Use the full training transform if provided
                    image = self.train_transform(image)
                else:
                    # Otherwise just use base transform (resize + convert to tensor, used for dataset creation)
                    # for human evaluation as we can't really judge the quality of normalized images
                    image = self.base_transform(image)
            
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        label = self.labels[idx]
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

    # calculate class counts before augmentation
    class_counts = defaultdict(int)
    for _, label in original_dataset:
        class_counts[label] += 1

    if Config.PLOT_CLASS_DISTRIBUTION:
        plot_class_distribution(
            data=class_counts,
            title="Pre-augmentation class distribution",
            xlabel="Class",
            ylabel="Number of images",
            save_path=Config.PLOTS_DIR / "pre_augmentation_class_distribution.png"
        )

    # Calculate target counts for each class
    max_class_count = max(class_counts.values())
    target_counts = {}
    
    for label, count in class_counts.items():
        if count == max_class_count:
            # Don't augment the largest class
            target_counts[label] = count
        else:
            # For smaller classes, augment up to either max_class_count or the maximum possible augmentations
            max_possible_count = count * Config.MAX_AUGMENTATION_FACTOR_PER_IMAGE
            target_counts[label] = min(max_class_count, max_possible_count)
    
    if Config.PLOT_CLASS_DISTRIBUTION:
        plot_class_distribution(
            data=target_counts,
            title="Post-augmentation class distribution",
            xlabel="Class",
            ylabel="Number of images",
            save_path=Config.PLOTS_DIR / "post_augmentation_class_distribution.png"
        )

    # split weights are based on target distribution
    weights = np.array([target_counts[label] for _, label in original_dataset])
    weights = weights / weights.sum()
    
    # stratified splits with weighted sampling
    indices = list(range(len(original_dataset)))
    labels = [label for _, label in original_dataset]
    
    # 1st split: train vs val+test
    train_indices, valtest_indices = weighted_train_test_split(
        indices=indices,
        labels=labels,
        weights=weights,
        test_size=train_val_split,
        random_state=Config.SEED
    )
    
    # 2nd split: val vs test
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

        # delete existing split if it exists
        if split_dir.exists():
            shutil.rmtree(split_dir)
            logger.info(f"Deleted existing split directory {split_dir}")
        split_dir.mkdir(parents=True, exist_ok=True)
        
        for idx in split_indices:
            img_path = original_dataset.images[idx]
            species = original_dataset.df.iloc[idx]['species']
            species_dir = split_dir / species
            species_dir.mkdir(exist_ok=True)
            
            shutil.copy2(img_path, species_dir)
            
            if should_augment:
                current_label = labels[idx]
                
                # Skip if this is from the largest class
                if class_counts[current_label] == max_class_count:
                    continue
                
                # Calculate how many augmentations we need for this image
                augs_needed = min(
                    Config.MAX_AUGMENTATION_FACTOR_PER_IMAGE - 1,  # -1 because we already have the original
                    (target_counts[current_label] - class_counts[current_label]) // class_counts[current_label]
                )
                
                if augs_needed > 0:
                    image = Image.open(img_path).convert('RGB')
                    base_name = Path(img_path).stem
                    extension = Path(img_path).suffix
                    
                    for aug_idx in range(augs_needed):
                        aug = random.choice(get_augmentation_list())
                        aug_image = aug(image)
                        aug_path = species_dir / f"{base_name}_aug{aug_idx + 1}{extension}"
                        aug_image.save(aug_path)
    
    logger.info("Static splits created successfully!")

    return output_dir / 'train', output_dir / 'val', output_dir / 'test'

def load_prepared_datasets(train_dir: Path, val_dir: Path, test_dir: Path) -> tuple[MushroomDataset, MushroomDataset, MushroomDataset]:
    """Load pre-split datasets."""
    if Config.CALCULATE_DATASET_MEAN_STD:
        logger.info("Calculating dataset mean and std...")
        
        temp_dataset = MushroomDataset(
            train_dir, 
            transform=get_base_transforms()
        )
        
        loader = DataLoader(
            temp_dataset,
            batch_size=Config.BATCH_SIZE,
            num_workers=4,
            shuffle=False
        )
        
        mean = torch.zeros(3)
        squared = torch.zeros(3)
        pixel_count = 0
        
        # First pass - mean
        for images, _ in tqdm(loader, desc="Calculating mean"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            pixel_count += batch_samples
            
        mean = mean / pixel_count
        
        # Second pass - std
        for images, _ in tqdm(loader, desc="Calculating std"):
            batch_samples = images.size(0)
            images = images.view(batch_samples, images.size(1), -1)
            squared += ((images - mean.unsqueeze(-1))**2).sum([0,2])
            
        std = torch.sqrt(squared / (pixel_count * images.size(-1)))
        
        logger.info(f"Dataset mean: {mean.tolist()}")
        logger.info(f"Dataset std: {std.tolist()}")
        
        Config.DATASET_MEAN = mean.tolist()
        Config.DATASET_STD = std.tolist()
    else:
        logger.info("Using pre-calculated dataset statistics from config")
        
    train_dataset = MushroomDataset(train_dir, transform=get_training_transforms())
    val_dataset = MushroomDataset(val_dir, transform=get_training_transforms())
    test_dataset = MushroomDataset(test_dir, transform=get_training_transforms())
    
    return train_dataset, val_dataset, test_dataset