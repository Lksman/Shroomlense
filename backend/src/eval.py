import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

from src.utils import get_logger
from src.config import Config

logger = get_logger(__name__)

def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """Calculate top-k accuracy.
    
    Args:
        outputs: Model predictions (N, num_classes)
        targets: Ground truth labels (N,)
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy as percentage
    """
    with torch.no_grad():
        _, pred = outputs.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / targets.size(0)).item()

def weighted_f1(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate weighted F1 score.
    
    Args:
        outputs: Model predictions (N, num_classes)
        targets: Ground truth labels (N,)
    
    Returns:
        Weighted F1 score
    """
    predictions = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, predictions, average='weighted')

def save_confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor, 
                         class_names: list[str], save_dir: str | Path,
                         model_name: str) -> None:
    """Generate and save confusion matrix visualization.
    
    Args:
        outputs: Model predictions (N, num_classes)
        targets: Ground truth labels (N,)
        class_names: List of class names
        save_dir: Directory to save the plot
        model_name: Name of the model for the filename
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    predictions = outputs.argmax(dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    cm = confusion_matrix(targets, predictions)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 20))
    sns.set_theme(style="white")
    sns.heatmap(
        cm_normalized, 
        annot=False, 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={"shrink": .8}
    )
    plt.title('Normalized Confusion Matrix', pad=20, size=16)
    plt.xlabel('Predicted', size=14)
    plt.ylabel('True', size=14)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.tight_layout()
    plt.savefig(save_dir / f'confusion_matrix_{model_name}_{timestamp}.png')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_dir}/confusion_matrix_{model_name}_{timestamp}.png")


def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, model_name: str, plot_confusion_matrix: bool = True):
    """Evaluate model and save results."""
    model.eval()
    device = next(model.parameters()).device
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            
            all_outputs.append(outputs)
            all_targets.append(targets)
    
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    top1_acc = top_k_accuracy(all_outputs, all_targets, k=1)
    top3_acc = top_k_accuracy(all_outputs, all_targets, k=3)
    top5_acc = top_k_accuracy(all_outputs, all_targets, k=5)
    f1 = weighted_f1(all_outputs, all_targets)
    
    # Save confusion matrix to global plots directory
    if plot_confusion_matrix:
        save_confusion_matrix(
            all_outputs, 
            all_targets, 
            test_loader.dataset.classes,
            Config.PLOTS_DIR / 'confusion_matrices',
            model_name
        )
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'weighted_f1': f1
    }