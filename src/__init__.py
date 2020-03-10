from flask_restful import Api
from app import flaskAppInstance
from .omrEvaluateAPI import omrImageProcessing


restServerInstance = Api(flaskAppInstance)

restServerInstance.add_resource(omrImageProcessing,"/exam/v1/omrevaluate")