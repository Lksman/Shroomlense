from flask import Flask
from flask_restx import Api

from src.utils import get_logger, Config
from api.external import external_ns
from api.inference import inference_ns

logger = get_logger(__name__)


def create_app():
    logger.info("Creating the app")

    app = Flask(Config.API_TITLE)

    api = Api(
        app,
        version=Config.API_VERSION,
        title=Config.API_TITLE,
        description=Config.API_DESCRIPTION,
    )

    api.add_namespace(external_ns)
    api.add_namespace(inference_ns)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host=Config.HOST, port=Config.PORT)
