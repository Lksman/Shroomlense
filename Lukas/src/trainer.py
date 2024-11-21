from pathlib import Path
import torch
from torch.types import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import f1_score

from src.utils import get_logger
from src.config import Config

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
        
        self.save_dir = Config.PLOTS_DIR
        self.save_dir.mkdir(exist_ok=True)
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
        """Run one epoch of training.
        
        Returns:
            tuple: (epoch_loss, epoch_f1)
                - epoch_loss: Average loss over the epoch
                - epoch_f1: Macro F1 score for the epoch (percentage)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for images, labels in pbar:
            loss, outputs = self._step(images, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.to(self.device)).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/len(self.train_loader):.4f}',
                'f1': f'{100.*correct/total:.2f}%'
            })

        epoch_loss = total_loss / len(self.train_loader)
        epoch_f1 = f1_score(labels.cpu(), predicted.cpu(), average='macro') * 100
        self.metrics['train_losses'].append(epoch_loss)
        self.metrics['train_f1s'].append(epoch_f1)
        
        logger.info(f"Training - Loss: {epoch_loss:.4f}, F1: {epoch_f1:.2f}%")
        return epoch_loss, epoch_f1

    def validate(self) -> tuple[float, float]:
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

    def plot_metrics(self, model_name: str = 'model') -> None:
        """Plot and save training and validation metrics.
        
        Args:
            model_name: Name of the model for saving the plot
        """
        epochs = range(1, len(self.metrics['train_losses']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(epochs, self.metrics['train_losses'], 'b-', label='Training')
        ax1.plot(epochs, self.metrics['val_losses'], 'r-', label='Validation')
        ax1.set_title('Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy/F1 plot
        ax2.plot(epochs, self.metrics['train_f1s'], 'b-', label='Train F1')
        ax2.plot(epochs, self.metrics['val_f1s'], 'r-', label='Val F1')
        ax2.set_title('Metrics')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        save_path = self.save_dir / f'{model_name}_metrics.png'
        fig.savefig(save_path)
        plt.close()
        logger.info(f"Metrics plot saved to {save_path}")

    def train(self, num_epochs: int, model_name: str, start_epoch: int = 0) -> float:
        """Run the complete training loop.
        
        Args:
            num_epochs: Number of epochs to train for
            model_name: Name of the model (used for saving)
            start_epoch: Epoch to start from (for resuming training)
            
        Returns:
            float: Best validation F1 score achieved
        """
        best_val_f1 = 0
        model_path = Config.MODELS_DIR / f"best_{model_name}_model.pth"
        
        logger.info(f"Training model {model_name} for {num_epochs} epochs")
        for epoch in range(start_epoch, num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            train_loss, train_f1 = self.train_epoch()
            val_loss, val_f1 = self.validate()
            
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f} | Train F1: {train_f1:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val F1: {val_f1:.2f}%")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"New best validation F1: {val_f1:.2f}%")
            
            if Config.SAVE_CHECKPOINTS and (epoch + 1) % Config.CHECKPOINT_FREQ == 0:
                checkpoint_path = Path(Config.CHECKPOINT_DIR, f'checkpoint_{model_name}_epoch_{epoch+1}.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        self.plot_metrics(model_name=model_name)
        return best_val_f1