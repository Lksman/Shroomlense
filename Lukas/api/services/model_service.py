import torch

from src.config import Config
from src.utils import get_logger
from src.models import create_model

logger = get_logger(__name__)

class ModelService:
    def __init__(self, image_service):
        self.model = create_model(model_name=Config.INFERENCE_MODEL_NAME, num_classes=43)[0]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        state_dict = torch.load(
            Config.INFERENCE_MODEL_PATH,
            map_location=self.device,
            weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Get class mappings from image service
        _, self.idx_to_class = image_service.get_class_mapping()
        
        logger.info(f"Model {Config.INFERENCE_MODEL_NAME} loaded successfully, using device {self.device}. Ready for inference.")

    def predict(self, image: torch.Tensor, top_k: int = 5) -> dict:
        """
        Run inference on preprocessed image
        
        Args:
            image: Preprocessed image tensor [1, C, H, W]
            top_k: Number of top predictions to return
            
        Returns:
            dict: Prediction results with top k class IDs, names and confidences
        """
        try:
            with torch.no_grad():
                image = image.to(self.device)
                outputs = self.model(image)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                confidences, predicted_classes = torch.topk(probabilities, k=top_k, dim=1)
                
                predictions = [
                    {
                        'class_id': class_id.item(),
                        'class_name': self.idx_to_class[class_id.item()],
                        'confidence': float(conf)
                    }
                    for class_id, conf in zip(predicted_classes[0], confidences[0])
                ]
                
                return {
                    'predictions': predictions
                }
                
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise RuntimeError("Model inference failed")