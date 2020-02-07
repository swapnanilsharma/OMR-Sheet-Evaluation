from flask import Flask, jsonify, request, flash
from flask_cors import CORS
from datetime import datetime, date
import json
import cv2
import os
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from imutils.perspective import four_point_transform
import warnings
import math
from imutils import contours
from datetime import datetime
from random import randint
from base64 import decodestring, decodebytes
import sys
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app, origins='*', allow_headers = '*')

def getContoursFromImage(imagePath, blockNumber):
    # load the image and compute the ratio of the old height to the new height, clone it, and resize it
    image = cv2.imread(imagePath)
    orig = image.copy()

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(src=gray, ksize=(3, 3), sigmaX=10, sigmaY=10)
    #gray = cv2.Sobel(src=gray, ddepth=-1, dx=2, dy=2, ksize=3)
    #cv2.imshow(winname="Answer key section of the image", mat=imutils.resize(warped, height=800))
    #cv2.waitKey(0)
    edged = cv2.Canny(image=gray, threshold1=75, threshold2=200, L2gradient=True, apertureSize=7)

    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(image=edged.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) #return countour tuple
    cnts = imutils.grab_contours(cnts) # extract cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:100]

    screenCnt = []
    # loop over the contours
    for idx, cnt in enumerate(cnts):
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # if our approximated contour has four points, then we can assume that we have found our screen
        if len(screenCnt)>100:
            break
        elif len(approx) == 4:
            if idx == 0:
                screenCnt.append(approx)
            elif abs(cv2.contourArea(cnt)-cv2.contourArea(cnts[idx-1])) > cv2.contourArea(cnt)*0.1:
                screenCnt.append(approx)
            else:
                pass

    total = [cv2.contourArea(cnt) for cnt in screenCnt]
    norm_list = [float(i)/sum([i for i in total]) for i in [i for i in total]]
    for idx, norm in enumerate(norm_list):
        if idx==0:
            pass
        elif norm > max(norm_list)*0.90:
            del screenCnt[idx]

    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(image=gray, pts=screenCnt[blockNumber].reshape(4, 2))
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    #warped = cv2.threshold(src=warped, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #paper = four_point_transform(image=orig, pts=screenCnt[blockNumber].reshape(4, 2))
    return warped, screenCnt, gray

def findContoursFromQASection(matrix, b_no):
    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(image=matrix.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) #return countour tuple
    cnts = imutils.grab_contours(cnts) # extract cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt=[]
    
    # loop over the contours
    for idx, cnt in enumerate(cnts):
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(curve=cnt, epsilon=0.02*peri, closed=True)
        # if our approximated contour has four points, then we can assume that we have found our screen
        if len(screenCnt)>100:
            break
        elif len(approx) == 4:
            if idx==0:
                screenCnt.append(approx)
            else:
                screenCnt.append(approx)

    total = [cv2.contourArea(cnt) for cnt in screenCnt]
    norm_list = [float(i)/sum([i for i in total]) for i in [i for i in total]]
    for idx, norm in enumerate(norm_list):
        if idx==0:
            pass
        elif norm > max(norm_list)*0.90:
            del screenCnt[idx]

    x_cor_for_max_circle = [cor[0] for cor in screenCnt[0].reshape(4,2)]
    for contr in screenCnt[1:6]:
        x_cor_for_nth_circle = [cor[0] for cor in contr.reshape(4,2)]
        if max(x_cor_for_nth_circle) <= math.ceil((b_no+1)*max(x_cor_for_max_circle)/5) and \
           min(x_cor_for_nth_circle) >= math.ceil(b_no*max(x_cor_for_max_circle)/5):
           # apply the four point transform to obtain a top-down view of the original image
           warped = four_point_transform(image=matrix, pts=contr.reshape(4, 2))
           warped = warped[:, int(0.235*warped.shape[1]):]
        else:
            pass
    return warped  

def findMarkedCircles(warped_sub, b_no):
    # find contours in the thresholded image, then initialize the list of contours that correspond to questions
    warped_sub =  cv2.threshold(warped_sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(warped_sub.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    print(f'------>>>>>Totl contours inside the image:{len(cnts)}')
    print(f'warped_sub: {warped_sub.shape}')
    
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 10 and h >= 10 and ar >= 0.70 and ar <= 1.30:
            questionCnts.append(c)
    
    print(f'------>>>>>Total valid question contours:{len(questionCnts)}')
    # sort the question cofntours top-to-bottom, then initialize the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts[:120], method="top-to-bottom")[0]
    response_key = []

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        # sort the contours or the current question from left to right, then initialize the index of the bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        bubbled = None
        total = []
        yeyy = []
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            #print(f'{j}---{c.shape}')
            # construct a mask that reveals only the current "bubble" for the question
            mask = np.zeros(warped_sub.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
            #cv2.imshow("Original haha", imutils.resize(mask, height=800))
            #cv2.waitKey(0)
        
            # apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
            mask = cv2.bitwise_and(warped_sub, warped_sub, mask=mask)
            total.append((cv2.countNonZero(mask), j))
            yeyy.append(cv2.moments(c)['m01'])  ###'m01'  
        norm_list1 = [float(i)/sum([i[0] for i in total]) for i in [i[0] for i in total]]
        norm_list = [float(i)/sum(yeyy) for i in yeyy]
        print(f'{q+1}-------{[i[0] for i in total]}----{norm_list1}')
        if min(norm_list1) <= 0.20:
            bubbled = total[norm_list1.index(min(norm_list1))]
        else:
            bubbled = (None, None)
        if   bubbled[1]==None:
            resp=""
        elif bubbled[1]==0:
            resp="A"
        elif bubbled[1]==1:
            resp="B"
        elif bubbled[1]==2:
            resp="C"
        elif bubbled[1]==3:
            resp="D"
        response_key.append({"questionno" : str(b_no*30+q+1), "answer": resp})
        #cv2.waitKey(0)
    return response_key

def findMarkedCircles1(warped_sub, b_no):
    #warped_sub =  cv2.threshold(warped_sub, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # find contours in the thresholded image, then initialize the list of contours that correspond to questions
    cnts = cv2.findContours(warped_sub.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    #questionCnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:120]


    print(len(cnts))
    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 10 and h >= 10 and ar >= 0.70 and ar <= 1.30:
            questionCnts.append(c)
    print(len(questionCnts))
    
    # sort the question cofntours top-to-bottom, then initialize the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts[:120], method="top-to-bottom")[0]
    response_key = []
    #print(f'----->>>>questionCnts: {len(questionCnts)}', file=sys.stderr)
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        #print(f'----->>>>iter: {len(q+1)}', file=sys.stderr)
        # sort the contours or the current question from left to right, then initialize the index of the bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        bubbled = None
        total = []
        yeyy = []
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current "bubble" for the question
            mask = np.zeros(warped_sub.shape, dtype="uint8")
            #cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
            mask = cv2.bitwise_and(warped_sub, warped_sub, mask=mask)
            #print(mask)
            total.append((cv2.countNonZero(mask), j))
            yeyy.append(cv2.moments(c)['m01'])
            #temp.append((cv2.countNonZero(mask), j))
        
        print(f'----->>>>{q+1} Norm list: {total}', file=sys.stderr)
        norm_list1 = [float(i)/sum([i[0] for i in total]) for i in [i[0] for i in total]]
        norm_list = [float(i)/sum(yeyy) for i in yeyy]
        #print(f'----->>>>{q+1} Norm list: {temp}', file=sys.stderr)
        print(f'----->>>>{q+1} Norm list: {norm_list1}', file=sys.stderr)
        if min(norm_list) <= 0.233:
            bubbled = total[norm_list.index(max(norm_list))]
        else:
            bubbled = (None, None)

        if   bubbled[1]==None:
            resp=""
        elif bubbled[1]==0:
            resp="A"
        elif bubbled[1]==1:
            resp="B"
        elif bubbled[1]==2:
            resp="C"
        elif bubbled[1]==3:
            resp="D"
        response_key.append({"questionno" : str(b_no*30+q+1), "answer": resp})
        cv2.waitKey(0)
    return response_key

def distance_betw_contours(contourList, idx1, idx2):
    import math
    p1 = contourList[idx1].reshape(4,2)[0]
    p2 = contourList[idx2].reshape(4,2)[0]
    return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )
     

def find_roll_contour(contourList):
    res = []
    temp = float('Inf')
    for idx, cntr in enumerate(contourList):
        if idx!=0 and distance_betw_contours(contourList=contourList, idx1=0, idx2=idx) < temp:
            res = idx
            temp = distance_betw_contours(contourList=contourList, idx1=0, idx2=idx)
    return res

def concatenate_list_data(list):
    result= ''
    for element in list:
        result += str(element)
    return result

def findRollNumber1(contourList, grayImage):
    roll_block = find_roll_contour(contourList=contourList)
    warped = four_point_transform(grayImage, contourList[roll_block].reshape(4, 2))
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    warped = warped[int(warped.shape[0]/5.1):, int(warped.shape[1]/11):]
    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    cnts = cv2.findContours(warped.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    print(f'Total cnts in Roll Number: {len(cnts)}')
    questionCnts = []
    for c in cnts:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 10 and h >= 10 and ar >= 0.70 and ar <= 1.30:
            questionCnts.append(c)
    print(f'Total questionCnts in Roll Number: {len(questionCnts)}')
    questionCnts = contours.sort_contours(questionCnts[:100], method="top-to-bottom")[0] #
    response_key = []
    
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 10)):
        cnts = contours.sort_contours(questionCnts[i:i + 10])[0]
        bubbled = None
        total = []
        yeyy = []
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            mask = np.zeros(warped.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
            cv2.imshow("Original haha", imutils.resize(mask, height=400))
            #cv2.imshow("Original haha thats it", imutils.resize(warped_sub, height=800))
            cv2.waitKey(0)
            # apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
            mask = cv2.bitwise_and(warped, warped, mask=mask)
            total.append((cv2.countNonZero(mask), j))
            yeyy.append(cv2.moments(c)['m01'])
            
        norm_list = [float(i)/sum(yeyy) for i in yeyy]
        print(f'Roll Number:-----------{norm_list}', file=sys.stderr)
        if min(norm_list) <= 0.093:
            bubbled = total[norm_list.index(min(norm_list))]
        else:
            bubbled = (-1, -1)
        response_key.append({str(q): (9 -bubbled[1])})
    
    if -1 in [list(i.values())[0] for i in response_key]:
        return 'error'
    else:
        return concatenate_list_data([list(i.values())[0] for i in response_key])

def findRollNumber(contourList, grayImage):
    roll_block = find_roll_contour(contourList=contourList)
    warped = four_point_transform(grayImage, contourList[roll_block].reshape(4, 2))
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    warped = warped[int(warped.shape[0]/5):, int(warped.shape[1]/11):]
    warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

    cnts = cv2.findContours(warped.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    for c in cnts:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 8 and h >= 8 and ar >= 0.60 and ar <= 1.40:
            questionCnts.append(c)
        
    questionCnts = contours.sort_contours(questionCnts[:100], method="top-to-bottom")[0]
    response_key = []
    
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 10)):
        cnts = contours.sort_contours(questionCnts[i:i + 10])[0]
        bubbled = None
        total = []
        yeyy = []
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            mask = np.zeros(warped.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
            # apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
            mask = cv2.bitwise_and(warped, warped, mask=mask)
            total.append((cv2.countNonZero(mask), j))
            yeyy.append(cv2.moments(c)['m01'])
            
        norm_list = [float(i)/sum(yeyy) for i in yeyy]
        print(f'Roll Number:-----------{norm_list}', file=sys.stderr)
        if min(norm_list) <= 0.093:
            bubbled = total[norm_list.index(min(norm_list))]
        else:
            bubbled = (-1, -1)
        response_key.append({str(q): (9 -bubbled[1])})
    
    if -1 in [list(i.values())[0] for i in response_key]:
        return 'error'
    else:
        return concatenate_list_data([list(i.values())[0] for i in response_key])


def processOMR(imagePath, blockNumber=1, answerKey=True, noOfQuestions=150):
    print(answerKey)
    #   resp, res_string, res, error_index
    if answerKey=='False': #'True'
        warped, screenCnt = getContoursFromImage(imagePath=imagePath, blockNumber=blockNumber)
        finalList = []
        for b_no in range(0,5,1):
            warped1 = findContoursFromQASection(matrix=warped, b_no=b_no)
            finalList = finalList + findMarkedCircles(warped_sub=warped1, b_no=b_no)
        finalList=finalList[0:noOfQuestions]
        error_index=[]
        for i in range(int(noOfQuestions)):
            if list(finalList[i].values())[0]=="":
                error_index.append(i+1)
        for i in range(int(noOfQuestions)):
            if len(error_index) != 0:
                return None, "answers keys are not proper", "FAILED", error_index, ""
        return finalList, "OK", "OK", None, ""
    elif answerKey=='True': #'False'
        warped, screenCnt, gray = getContoursFromImage(imagePath=imagePath, blockNumber=blockNumber)
        #print(f'----->>>>screenCnt:{len(screenCnt)}', file=sys.stderr)
        roll_no = findRollNumber(contourList=screenCnt, grayImage=gray)
        #print(f'----->>>>roll no:{roll_no}', file=sys.stderr)
        finalList = []
        for b_no in range(0,5,1):
            warped1 = findContoursFromQASection(matrix=warped, b_no=b_no)
            finalList = finalList + findMarkedCircles(warped_sub=warped1, b_no=b_no)
        finalList=finalList[0:noOfQuestions]
        return finalList,  "OK", "OK", None, roll_no

@app.route('/exam/v1/omrevaluate',  methods=['POST'])
def omrevaluate():

    noOfQuestions=request.form.get("noofquestions") #or request.get_json().get('noofquestions', '')
    answerKey=request.form.get('answerkey') #or request.get_json().get('answerkey', '')
    templateId=request.form.get('templateid') #or request.get_json().get('answerkey', '')
    '''
    image= request.form.get('imagebase64')
    imagePath="imageToSave.jpg"
    with open(imagePath, "wb") as fh:
        fh.write(decodebytes(image.encode()))
    '''

    image = request.files['imagefile']
    image.save(image.filename)
    imagePath = image.filename

    blockNumber = 1
    #imagePath = 'C:/Users/Swapnanil/Machine Learning/PwC/swap102.jpg' #200
    #try:
    resp, res_string, res, error_index, roll_no = processOMR(imagePath=imagePath, blockNumber=blockNumber, answerKey=answerKey, noOfQuestions=int(noOfQuestions))
    '''
    except:
        return jsonify({"result": {"status": "FAILED"},
                  "student": {"class":"",
                              "section":"",
                              "schoolcode":"",
                              "rollno":""},
                  "answers": [],
                  "params": {"msgid": None,
                             "resmsgid": None,
                             "err": None,
                             "err_msg": None,
                             "err_detail": None,
                             "status": 'rescan the image'},
                  "responseCode": 'FAILED'
                  })
    '''

    if os.path.exists(imagePath):
        pass
        #os.remove(imagePath)
    print(error_index)
    pars = []
    try:
        for i in error_index:
            pars.append({"msgid": None,
                         "resmsgid": None,
                         "err": str(i),
                         "err_msg": 'Question no:'+str(i)+' is not proper',
                         "err_detail": None,
                         "status": 'FAILED'})
    except:
        pars=[]
    temp=randint(10000, 99999)
    return jsonify({"result": {"status": "OK"},
                    "student": {"name": "student_"+str(temp),
                              "class": "",
                              "section": "",
                              "schoolcode": "",
                              "rollno": roll_no},
                  "answers": resp,
                  "params": pars,
                  "responseCode": res
                  })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8999, debug=True)