from PIL import Image
import numpy as np
import io
import torch
import random
from torchvision import transforms
from typing import Tuple, Dict

from src.config import Config
from src.utils import get_logger
from src.dataset import MushroomDataset

logger = get_logger(__name__)

class ImageService:
    def __init__(self):
        """Initialize the image service with transforms and dataset mappings."""
        self.transform = transforms.Compose([
            transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.DATASET_MEAN, std=Config.DATASET_STD)
        ])
        
        self.dataset = MushroomDataset(Config.DATA_DIR, skip_transform=True)
        self.class_to_idx = self.dataset.class_to_idx
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}
        
        # Create mapping of class names to their image paths
        self.class_to_images = self.dataset.species_paths
        
        # case-insensitive, flexible with spaces/underscores mappings
        self.class_names_lower = {
            self._normalize_name(cls_name): cls_name 
            for cls_name in self.class_to_idx.keys()
        }
        
    def _normalize_name(self, name: str) -> str:
        """
        Normalize class name by converting to lowercase and replacing spaces/underscores.
        
        Args:
            name: The class name to normalize
            
        Returns:
            Normalized class name string
        """
        return name.lower().replace(' ', '_').replace('-', '_')
        
    def _validate_class_name(self, class_name: str) -> str:
        """
        Validate and return the correct class name.
        
        Args:
            class_name: The class name to validate
            
        Returns:
            The correctly formatted class name
            
        Raises:
            ValueError: If the class name is invalid
        """
        normalized_name = self._normalize_name(class_name)
        if normalized_name not in self.class_names_lower:
            available_classes = ', '.join(sorted(self.class_names_lower.values()))
            raise ValueError(
                f"Invalid class name: {class_name}. "
                f"Available classes: {available_classes}"
            )
        return self.class_names_lower[normalized_name]
        
    def get_class_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        Return the class mappings.
        
        Returns:
            Tuple containing:
            - Dict mapping class names to indices
            - Dict mapping indices to class names
        """
        return self.class_to_idx, self.idx_to_class
        
    def get_random_image_by_class(self, class_name: str) -> Tuple[Image.Image, str]:
        """
        Get a random image from a specific class (case insensitive, flexible with spaces/underscores).
        
        Args:
            class_name: Name of the mushroom class
            
        Returns:
            Tuple containing:
            - PIL Image object
            - Path to the image file
            
        Raises:
            ValueError: If the class name is invalid
            IOError: If there's an error loading the image
        """
        correct_class_name = self._validate_class_name(class_name)
        image_path = random.choice(self.class_to_images[correct_class_name])
        
        try:
            image = Image.open(image_path).convert('RGB')
            return image, image_path
        except IOError as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise IOError(f"Failed to load image for class {class_name}")
        
    def preprocess_image(self, image_file) -> torch.Tensor:
        """
        Preprocess image from API request for model inference.
        
        Args:
            image_file: FileStorage object from Flask request
            
        Returns:
            Preprocessed image tensor [1, C, H, W]
            
        Raises:
            ValueError: If the image format is invalid or file is corrupted
        """
        try:
            # image from bytes
            image_bytes = image_file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image)
            return image_tensor.unsqueeze(0) # adds batch dimension

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError("Invalid image format or corrupted file")

