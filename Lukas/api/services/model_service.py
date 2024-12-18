# TODO: implement model service
# should be able to load a model from the weights (from the models directory), predict on an image, and return the prediction
# the model should be instantiated at import time, so that it can be used as a singleton
# also sanitize the input and check the types etc etc.

import numpy as np

from src.config import Config
from src.utils import get_logger
from src.models import create_model

logger = get_logger(__name__)

class ModelService:
    def __init__(self):
        self.model = create_model(model_name=Config.INFERENCE_MODEL_NAME, num_classes=43)

    def predict(self, image: np.ndarray) -> str:
        # TODO: implement prediction: preprocess the image, run the model, return the prediction
        # TODO: sanitize the input and check the types etc etc.

        pass

model_service = ModelService()
