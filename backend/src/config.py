from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

class Config:
    # Data paths relative to project root
    DATA_DIR = PROJECT_ROOT.parent / "data/mushroom_50k_v1"
    MODELS_DIR = PROJECT_ROOT / "models"
    SERIALIZED_MODELS_DIR = PROJECT_ROOT / "api/serialized_models"
    PLOTS_DIR = PROJECT_ROOT / "plots"
    INTERMEDIATE_DATA_DIR = PROJECT_ROOT.parent / "data/intermediate"
    
    # Create directories
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    INTERMEDIATE_DATA_DIR.mkdir(exist_ok=True, parents=True)

    # General settings
    SEED = 42


    # Image parameters
    IMAGE_SIZE = 224  # input size of vgg19    
    CALCULATE_DATASET_MEAN_STD = False # set to true if we change the dataset, the current mean and std are calculated from the unaugmented mushroom_50k_v1 dataset
    DATASET_MEAN = [0.4519599378108978, 0.43002936244010925, 0.3518902063369751]
    DATASET_STD = [0.25967419147491455, 0.24217985570430756, 0.24647024273872375]
    # IMAGENET_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    # IMAGENET_STD = [0.229, 0.224, 0.225]


    # Dataset settings
    MAX_AUGMENTATION_FACTOR_PER_IMAGE = 5
    CREATE_STATIC_SPLITS = False # set to true to (re)create static splits, false = use existing ones in the intermediate data directory
    PLOT_CLASS_DISTRIBUTION = True # set to true to plot the class distribution before and after augmentation
    BATCH_SIZE = 32 # runs fine on my (lukas) machine. if you run out of memory, reduce this to 16 or 8

    # General model parameters
    NUM_CLASSES = None  # Will be set dynamically based on dataset
    PIPELINE_TESTING_MODE = False # set to true to run the pipeline in testing mode (i.e. train for 0 epochs), false = run the full training pipeline
    NUM_EPOCHS = {
        'cnn': 20,
        'vgg19': 5,
        'vit': 5
    } # will be ignored if PIPELINE_TESTING_MODE is true
    
    # Model configurations
    MODELS_TO_TRAIN = [
        # 'cnn', 
        # 'vgg19', 
        'vit'
    ]
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

    # Plotting settings
    PLOT_METRICS = True
    PLOT_CONFUSION_MATRIX = True


    # Logging settings
    LOG_LEVEL = "DEBUG"
    LOG_DIR = PROJECT_ROOT / "logs"
    
    # API settings
    API_TITLE = "Shroomlense API"
    API_VERSION = "0.2"
    API_DESCRIPTION = "API for mushroom species classification"
    HOST = "localhost"
    PORT = 5000

    INFERENCE_MODEL_NAME = "vit"
    INFERENCE_MODEL_PATH = SERIALIZED_MODELS_DIR / INFERENCE_MODEL_NAME / "best_model.pth"

    CUMULATIVE_CONFIDENCE_WARNING_THRESHOLD = 0.9
    TOP_1_CONFIDENCE_WARNING_THRESHOLD = 0.6