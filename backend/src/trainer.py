from pathlib import Path
import torch
from torch.types import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import f1_score
import seaborn as sns

from src.utils import get_logger, save_model
from src.config import Config
from src.dataset import MushroomDataset
from src.eval import top_k_accuracy

logger = get_logger(__name__)

class Trainer:
    """Trainer class for handling the training loop of a neural network."""
    
    def __init__(self, 
                 model: nn.Module, 
                 train_loader: DataLoader, 
                 val_loader: DataLoader, 
                 optimizer: Optimizer, 
                 device: str, 
                 scheduler: Optimizer = None):
        """Initialize the trainer.
        
        Args:
            model: Neural network model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for updating model parameters
            device: Device to run training on ('cuda' or 'cpu')
            scheduler: Optional learning rate scheduler
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        
        self.metrics = {
            'train_losses': [],
            'val_losses': [],
            'train_f1s': [],
            'val_f1s': []
        }
        
        logger.info(f"Training on device: {device}")

    def _step(self, images: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
        """Perform a single training step.
        
        Args:
            images: Input image batch tensor (shape: B x C x H x W)
            labels: Ground truth label tensor (shape: B)
            
        Returns:
            tuple: (loss, outputs)
                - loss: Scalar loss tensor
                - outputs: Model predictions tensor (shape: B x num_classes)
        """
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()
        
        return loss, outputs

    def train_epoch(self) -> tuple[float, float]:
        """Run one epoch of training."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            loss, outputs = self._step(images, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(self.device)).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(self.train_loader):.4f}',
                'acc': f'{100.*correct/total:.2f}%'  # using acc during the epoch since F1 needs full epoch
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_f1 = f1_score(all_labels, all_predictions, average='macro') * 100
        self.metrics['train_losses'].append(epoch_loss)
        self.metrics['train_f1s'].append(epoch_f1)
        
        return epoch_loss, epoch_f1

    def validate(self) -> tuple[float, float, float, float]:
        """Run validation on the validation set.
        
        Returns:
            tuple: (epoch_loss, val_f1)
                - epoch_loss: Average validation loss
                - val_f1: Macro F1 score on validation set (percentage)
        """
        self.model.eval()
        total_loss = 0
        predictions, ground_truth = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc='Validating'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                ground_truth.extend(labels.cpu().numpy())
        
        epoch_loss = total_loss / len(self.val_loader)
        val_f1 = f1_score(ground_truth, predictions, average='macro') * 100
        
        self.metrics['val_losses'].append(epoch_loss)
        self.metrics['val_f1s'].append(val_f1)
        return epoch_loss, val_f1

    def plot_metrics(self, model_name: str = 'model', save_dir: Path = None) -> None:
        """Plot and save training and validation metrics."""
        sns.set_style("whitegrid")
        sns.set_palette("deep")
        
        epochs = range(1, len(self.metrics['train_losses']) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Loss plot
        sns.lineplot(x=epochs, y=self.metrics['train_losses'], label='Training', ax=ax1)
        sns.lineplot(x=epochs, y=self.metrics['val_losses'], label='Validation', ax=ax1)
        ax1.set_title('Loss', pad=10)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        # F1 Score plot
        sns.lineplot(x=epochs, y=self.metrics['train_f1s'], label='Train F1', ax=ax2)
        sns.lineplot(x=epochs, y=self.metrics['val_f1s'], label='Val F1', ax=ax2)
        ax2.set_title('F1 Score', pad=10)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score (%)')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        plt.tight_layout()
        save_path = save_dir / f'{model_name}_metrics.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Metrics plot saved to {save_path}")

    def train(self, num_epochs: int, model_name: str, save_dir: Path) -> tuple[Path, dict]:
        """Run the complete training loop.
        
        Args:
            num_epochs: Number of epochs to train for (0 for pipeline testing)
            model_name: Name of the model (used for saving)
            save_dir: Directory to save model and plots
            
        Returns:
            tuple: (best_model_path, best_metrics)
        """
        if Config.PIPELINE_TESTING_MODE:
            logger.info(f"Pipeline test mode: Saving initial model state without training")
            initial_metrics = {
                'val_f1': 0,
                'val_loss': 0,
                'train_f1': 0,
                'train_loss': 0,
                'epoch': 0,
                'pipeline_test': True
            }
            model_path = save_model(self.model, model_name, initial_metrics, save_dir)
            return model_path, initial_metrics
            
        best_val_f1 = 0
        best_model_path = None
        best_metrics = None
        
        logger.info(f"Training model {model_name} for {num_epochs} epochs")
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_f1 = self.train_epoch()
            val_loss, val_f1 = self.validate()
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.2f}%")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_metrics = {
                    'val_f1': val_f1,
                    'val_loss': val_loss,
                    'train_f1': train_f1,
                    'train_loss': train_loss,
                    'epoch': epoch + 1
                }
                best_model_path = save_model(self.model, model_name, best_metrics, save_dir)
                logger.info(f"New best validation F1: {val_f1:.2f}%")
        
        if Config.PLOT_METRICS:
            self.plot_metrics(model_name=model_name, save_dir=save_dir)
            
        return best_model_path, best_metrics
    

def setup_training(model: torch.nn.Module, config: dict, train_dataset: MushroomDataset, val_dataset: MushroomDataset) -> Trainer:
    """Setup training components."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    

    # TODO: AdamW for now (faster convergence), switch to SGD for fine-tuning
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=config['learning_rate'],
    #     weight_decay=config['weight_decay']
    # )

    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )
    
    return trainer