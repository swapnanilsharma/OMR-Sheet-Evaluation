from flask import request, jsonify
from flask_restful import Resource
import logging as logger
import json
import os
import warnings

from .cornerFeatures import cornerFeatures
from .omrServiceTemplate1 import omrServiceTemplate1
from .responseJson import *
from .commonConstants import *

warnings.filterwarnings("ignore")


class omrImageProcessing(Resource):

    def post(self):
        noOfQuestions = request.form.get(NO_OF_QUESTIONS)
        answerKey = request.form.get(ANSWER_KEY)
        templateId = request.form.get(TEMPLATE_ID)

        image = request.files[IMAGE_FILE]
        image.save(image.filename)
        imagePath = image.filename

        try:
            if int(noOfQuestions) > MAX_QUES_TEMP1:
                raise
            omrServObj = omrServiceTemplate1()
            answerResponse, responseCode, errIndex = omrServObj.processOMR(imagePath=imagePath,
                                                                           answerKey=answerKey,
                                                                           noOfQuestions=int(noOfQuestions))
            if answerKey == FALSE:
                rollNumber, schoolCode, studentName, class_, section = omrServObj.otherDetailsOfStudent(imagePath=imagePath)
            else:
                rollNumber, schoolCode, studentName, class_, section = EMPTY, EMPTY, EMPTY, EMPTY, EMPTY
            student = studentJson(name=studentName, class_=class_, section=section, schoolcode=schoolCode, rollno=rollNumber)

        except:
            params = paramsJson(msgid=EMPTY, resmsgid=EMPTY, err=EMPTY,
                                err_msg=EMPTY, err_detail=EMPTY,
                                status=RESCAN_ERR_MSG)
            student = studentJson(name=EMPTY, class_=EMPTY,
                                  section=EMPTY, schoolcode=EMPTY,
                                  rollno=EMPTY)
            answers = answersJson([])
            return json.loads(finalRespJson(result=FAILED, params=params,
                                            responseCode=FAILED,
                                            student=student,
                                            answers=answers).getJson)

        if os.path.exists(imagePath):
            # pass
            os.remove(imagePath)
        pars = []
        if errIndex:
            params = paramsJson(msgid=EMPTY, resmsgid=EMPTY, err=errIndex,
                                err_msg=QUES_NO_NOT_PROP_ERR_MSG,
                                err_detail=EMPTY, status=FAILED)
        else:
            params = paramsJson(msgid=EMPTY, resmsgid=EMPTY,
                                err=EMPTY, err_msg=EMPTY,
                                err_detail=EMPTY, status=OK)
        answers = answersJson(answerResponse)
        return json.loads(finalRespJson(result=OK, params=params,
                                        responseCode=responseCode,
                                        student=student, answers=answers).getJson)
