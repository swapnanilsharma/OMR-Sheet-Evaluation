from flask_restful import Api
from app import flaskAppInstance
from .omrEvaluateAPI import omrImageProcessing
from .config import API_ENDPOINT


restServerInstance = Api(flaskAppInstance)

restServerInstance.add_resource(omrImageProcessing, API_ENDPOINT)