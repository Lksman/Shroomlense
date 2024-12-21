import random
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config
from src.dataset import create_static_splits, load_prepared_datasets
from src.trainer import setup_training
from src.models import create_model
from src.utils import get_logger
from src.eval import evaluate_model

logger = get_logger(__name__)

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(Config.MODELS_DIR, timestamp)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    random.seed(Config.SEED)
    
    if Config.CREATE_STATIC_SPLITS:
        logger.info("Creating static splits...")
        train_dir, val_dir, test_dir = create_static_splits(
            data_dir=Config.DATA_DIR,
            output_dir=Config.INTERMEDIATE_DATA_DIR,
            train_val_split=0.2,
            val_test_split=0.5
        )
    else:
        logger.info("Using existing static splits...")
        train_dir = Config.INTERMEDIATE_DATA_DIR / 'train'
        val_dir = Config.INTERMEDIATE_DATA_DIR / 'val'
        test_dir = Config.INTERMEDIATE_DATA_DIR / 'test'

        if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
            raise ValueError("Static splits directory does not exist or is incomplete. Please set CREATE_STATIC_SPLITS to True to create them.")

    # Prepare datasets from static splits
    logger.info("Loading datasets from static splits...")
    train_dataset, val_dataset, test_dataset = load_prepared_datasets(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir
    )

    all_results = {}
    
    # Train and evaluate each model
    for model_name in Config.MODELS_TO_TRAIN:
        logger.info(f"\nTraining {model_name.upper()} model")
        model_dir = results_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create and train model
        model, config = create_model(model_name, Config.NUM_CLASSES)
        trainer = setup_training(model, config, train_dataset, val_dataset)
        
        # Train model and get best weights path and metrics
        best_model_path, train_metrics = trainer.train(
            num_epochs=Config.NUM_EPOCHS[model_name],
            model_name=model_name,
            save_dir=model_dir
        )
        
        # Load best model weights for evaluation
        model.load_state_dict(torch.load(
            best_model_path,
            map_location=trainer.device,
            weights_only=True
        ))
        model.eval()
        
        # Setup test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available()
        )
        
        # Evaluate model
        logger.info(f"Evaluating {model_name} model on test set")
        metrics = evaluate_model(model, test_loader, model_name, plot_confusion_matrix=Config.PLOT_CONFUSION_MATRIX)
        all_results[model_name] = metrics
        
        # Log results
        logger.info(f"\n{model_name.upper()} Results:")
        logger.info(f"Model saved at: {best_model_path}")
        logger.info(f"Top-1 Accuracy: {metrics['top1_accuracy']:.2f}%")
        logger.info(f"Top-3 Accuracy: {metrics['top3_accuracy']:.2f}%")
        logger.info(f"Top-5 Accuracy: {metrics['top5_accuracy']:.2f}%")
        logger.info(f"Weighted F1 Score: {metrics['weighted_f1']:.4f}")
    
    # Save all results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_path = Config.PLOTS_DIR / 'evaluations'
    eval_path.mkdir(exist_ok=True)
    
    with open(eval_path / f'evaluation_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()