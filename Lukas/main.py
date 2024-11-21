import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from src.config import Config
from src.dataset import prepare_datasets
from src.trainer import Trainer
from src.models import MushroomCNN, MushroomVGG19
from src.utils import get_logger, load_checkpoint

logger = get_logger(__name__)

def parse_cli_args():
    parser = argparse.ArgumentParser(description='Train mushroom classifier')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS, help='number of epochs to train')
    parser.add_argument('--model', type=str, default=Config.MODEL_TYPE, choices=['vgg', 'cnn'], help='model architecture to use')
    return parser.parse_args()

def main():    
    args = parse_cli_args()
    
    logger.info(f"Preparing datasets from {Config.DATA_DIR}")
    train_dataset, val_dataset = prepare_datasets(Config.DATA_DIR)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    if args.model == 'vgg':
        logger.info("Using VGG19 model")
        model = MushroomVGG19(Config.NUM_CLASSES).to(Config.DEVICE)
        learning_rate = Config.VGG_LR
    else:
        logger.info("Using CNN model")
        model = MushroomCNN(Config.NUM_CLASSES).to(Config.DEVICE)
        learning_rate = Config.CNN_LR
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint_path = Path(Config.CHECKPOINT_DIR, args.resume)
        if load_checkpoint(model, checkpoint_path):
            try:
                start_epoch = int(Path(args.resume).stem.split('_')[-1])
                logger.info(f"Resuming from epoch {start_epoch}")
            except:
                logger.warning("Could not determine start epoch from filename, starting from scratch")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=Config.DEVICE,
        scheduler=scheduler
    )
    
    trainer.train(
        num_epochs=args.epochs,
        model_name=args.model,
        start_epoch=start_epoch
    )

if __name__ == "__main__":
    main() 