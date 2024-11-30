from flask import request
from flask_restx import Resource, Namespace
from werkzeug.datastructures import FileStorage

from src.utils import get_logger

logger = get_logger(__name__)

inference_ns = Namespace("inference", description="Inference operations")

@inference_ns.route("/predict")
@inference_ns.expect(inference_ns.parser().add_argument('image', location='files', type=FileStorage, required=True))
class Predict(Resource):
    def post(self):
        if 'image' not in request.files:
            logger.error("No image provided")
            return {'error': 'No image provided'}, 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("No selected file")
            return {'error': 'No selected file'}, 400

        logger.info("Image received successfully")    
        # TODO: further process image 
        
        return {'message': 'Image received successfully'}, 200
