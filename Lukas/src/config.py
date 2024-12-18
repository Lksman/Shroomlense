from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent

class Config:
    # Data paths relative to project root
    DATA_DIR = PROJECT_ROOT.parent / "data/mushroom_50k_v1"
    MODELS_DIR = PROJECT_ROOT / "models"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    INTERMEDIATE_DATA_DIR = PROJECT_ROOT.parent / "data/intermediate"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    INTERMEDIATE_DATA_DIR.mkdir(exist_ok=True, parents=True)

    # General settings
    SEED = 42

    # Dataset settings
    MAX_AUGMENTATION_FACTOR_PER_IMAGE = 5
    AUGMENTATION_STEPS = {
        "rotation": True,
        "color_jitter": True,
        "affine": False
    }
    CREATE_STATIC_SPLITS = False # set to true to (re)create static splits, false = use existing ones in the intermediate data directory


    # General model parameters
    NUM_CLASSES = None  # Will be set dynamically based on dataset
    BATCH_SIZE = 32 # runs fine on my (lukas) machine. if you run out of memory, reduce this to 16 or 8
    NUM_EPOCHS = 0 # for testing the pipeline set to 0, set to 20 for full training
    
    # Model configurations
    MODELS_TO_TRAIN = ['cnn', 'vgg19', 'vit']
    MODEL_CONFIGS = {
        'cnn': {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
        },
        'vgg19': {
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'batch_size': 16,
        },
        'vit': {
            'learning_rate': 5e-5,
            'weight_decay': 1e-4,
            'batch_size': 16,
        }
    }

    
    # Image parameters
    IMAGE_SIZE = 224  # input size of vgg19    
    # IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    # IMAGENET_STD = [0.229, 0.224, 0.225]
    CALCULATE_DATASET_MEAN_STD = False # set to true if we change the dataset, the current mean and std are calculated from the unaugmented mushroom_50k_v1 dataset
    DATASET_MEAN = [-0.2131882607936859, -0.16706067323684692, -0.3030257225036621]
    DATASET_STD = [1.0976533889770508, 1.052354335784912, 1.0617069005966187]

    # Logging settings
    LOG_LEVEL = "DEBUG"
    LOG_DIR = Path("logs")
    
    # API settings
    INFERENCE_MODEL_NAME = "vit"
    API_TITLE = "Mushroom API"
    API_VERSION = "0.1"
    API_DESCRIPTION = "API for mushroom species classification"
    HOST = "0.0.0.0"
    PORT = 5000


