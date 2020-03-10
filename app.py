from flask import Flask
import logging as logger
from flask_cors import CORS
logger.basicConfig(level="DEBUG")

flaskAppInstance = Flask(__name__)
CORS(flaskAppInstance, origins='*', allow_headers='*')

if __name__ == '__main__':
    logger.debug("Starting Flask Server")
    from src import *
    flaskAppInstance.run(host="0.0.0.0", port=8999, debug=True, use_reloader=True)