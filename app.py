from flask import Flask
import logging as logger
from flask_cors import CORS
from src.config import HOST, PORT
logger.basicConfig(level="DEBUG")

flaskAppInstance = Flask(__name__)
CORS(flaskAppInstance, origins='*', allow_headers='*')

if __name__ == '__main__':
    logger.debug("------------------Starting Flask Server------------------")
    from src import *
    flaskAppInstance.run(host=HOST, port=PORT, debug=True, use_reloader=True)
