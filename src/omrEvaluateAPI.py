from flask import request, jsonify
from flask_restful import Resource
import logging as logger
import json
import os
import warnings
warnings.filterwarnings("ignore")

from .cornerFeatures import cornerFeatures
from .omrServiceTemplate1 import omrServiceTemplate1
from .responseJson import *

class omrImageProcessing(Resource):

    def post(self):
        noOfQuestions = request.form.get("noofquestions")
        answerKey = request.form.get('answerkey')
        templateId = request.form.get('templateid')
    
        image = request.files['imagefile']
        image.save(image.filename)
        imagePath = image.filename
    
        try:
            if int(noOfQuestions) > 150:
                raise
            omrServObj = omrServiceTemplate1()
            answerResponse, responseCode, error_index = omrServObj.processOMR(imagePath=imagePath,
                                                            answerKey=answerKey, noOfQuestions=int(noOfQuestions))
            if answerKey=='False':
                rollNumber, schoolCode, studentName, class_, section = omrServObj.otherDetailsOfStudent(imagePath=imagePath)
            else:
                rollNumber, schoolCode, studentName, class_, section = "", "", "", "", ""
            student = studentJson(name=studentName, class_=class_, section=section, schoolcode=schoolCode, rollno=rollNumber)
    
        except:
            params = paramsJson(msgid="", resmsgid="", err="", err_msg="", 
                                err_detail="", status='EROOR!!! Rescan the image')
            student = studentJson(name="", class_="", section="", schoolcode="", rollno="")
            answers = answersJson([])
            return json.loads(finalRespJson(result="FAILED", params=params, responseCode="FAILED", 
                                            student=student, answers=answers).getJson())

    
        if os.path.exists(imagePath):
            #pass
            os.remove(imagePath)
        pars = []
        if error_index:
            params = paramsJson(msgid="", resmsgid="", err=str(error_index), err_msg='Question no:'+ str(error_index)+' are not proper', 
                       err_detail="", status='FAILED')
        else:
            params = paramsJson(msgid="", resmsgid="", err="", err_msg="", 
                                err_detail="", status='OK')
        answers = answersJson(answerResponse)
        return json.loads(finalRespJson(result="OK", params=params, responseCode=responseCode, 
                                        student=student, answers=answers).getJson())