import json

class responseJson:
    pass

class finalRespJson(responseJson):
    def __init__(self, result, student, answers, params, responseCode):
        self.result = resultJson(status=result).getJson()
        self.student = student.getJson()
        self.answers = answers.getJson()
        self.params = params.getJson()
        self.responseCode = responseCodeJson(responseCode).getJson()

    @property
    def getJson(self):
        return json.dumps(self.__dict__)

class resultJson(responseJson):
    def __init__(self, status):
        self.status = status

    def getJson(self):
        return self.__dict__

class studentJson(responseJson):
    def __init__(self, name, class_, section, schoolcode, rollno):
        self.name = name
        self.classs = class_
        self.section = section
        self.schoolcode = schoolcode
        self.rollno = rollno

    def getJson(self):
        return self.__dict__

class answersJson(responseJson):
    def __init__(self, answers):
        self.answers = answers

    def getJson(self):
        return self.answers

class paramsJson(responseJson):
    def __init__(self, msgid, resmsgid, err, err_msg, err_detail, status):
        self.msgid = msgid
        self.resmsgid = resmsgid
        self.err = err
        self.err_msg = err_msg
        self.err_detail = err_detail
        self.status = status

    def getJson(self):
        return self.__dict__ 

class responseCodeJson(responseJson):
    def __init__(self, responseCode):
        self.responseCode = responseCode

    def getJson(self):
        return self.responseCode