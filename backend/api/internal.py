from flask import request, send_file
from flask_restx import Resource, Namespace
from werkzeug.datastructures import FileStorage
import io

from src.utils import get_logger
from api.services.model_service import ModelService
from api.services.image_service import ImageService

logger = get_logger(__name__)

internal_ns = Namespace("internal", description="Internal operations")
image_service = ImageService()
model_service = ModelService(image_service)

@internal_ns.route("/predict")
@internal_ns.expect(internal_ns.parser().add_argument('image', location='files', type=FileStorage, required=True))
class Predict(Resource):
    def post(self):
        if 'image' not in request.files:
            logger.error("No image provided")
            return {'error': 'No image provided'}, 400
            
        image_file = request.files['image']
        if image_file.filename == '':
            logger.error("No selected file")
            return {'error': 'No selected file'}, 400

        try:
            preprocessed_image = image_service.preprocess_image(image_file)
            result = model_service.predict(preprocessed_image, top_k=5)
            
            return result, 200
            
        except ValueError as e:
            return {'error': str(e)}, 400
        except RuntimeError as e:
            return {'error': str(e)}, 500
        except Exception as e:
            logger.error(f"Unexpected error during inference: {e}")
            return {'error': 'Internal server error'}, 500

@internal_ns.route("/random_image/<string:mushroom_name>")
class RandomImage(Resource):
    def get(self, mushroom_name):
        try:
            image, image_path = image_service.get_random_image_by_class(mushroom_name)
            img_io = io.BytesIO()
            image.save(img_io, 'JPEG')
            img_io.seek(0)
            
            return send_file(
                img_io,
                mimetype='image/jpeg',
                as_attachment=False,
                download_name=f"{mushroom_name}_random.jpg"
            )
            
        except ValueError as e:
            return {'error': str(e)}, 400
        except Exception as e:
            logger.error(f"Error getting random image: {e}")
            return {'error': 'Internal server error'}, 500
