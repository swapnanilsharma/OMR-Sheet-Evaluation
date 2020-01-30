# import the necessary packages
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
warnings.filterwarnings("ignore")


def getContoursFromImage(imagePath, blockNumber):
    # load the image and compute the ratio of the old height to the new height, clone it, and resize it
    image = cv2.imread(imagePath)
    orig = image.copy()

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
    edged = cv2.Canny(image=gray, threshold1=75, threshold2=200, L2gradient=False)

    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(image=edged.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) #return countour tuple
    cnts = imutils.grab_contours(cnts) # extract cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:200]

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
    warped = cv2.threshold(src=warped, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    paper = four_point_transform(image=orig, pts=screenCnt[blockNumber].reshape(4, 2))
    return warped, paper

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
        else:
            pass
    return warped

def findContoursFromQAColumn(matrix, sb_no):
    return matrix[(sb_no)*int(matrix.shape[0]/6):(sb_no+1)*int(matrix.shape[0]/6),0:matrix.shape[1]]
    

def findMarkedCircles(warped_sub, b_no, sb_no):
    # find contours in the thresholded image, then initialize the list of contours that correspond to questions
    cnts = cv2.findContours(warped_sub.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 30 and h >= 30 and ar >= 0.90 and ar <= 1.2:
            questionCnts.append(c)
    # sort the question cofntours top-to-bottom, then initialize the total number of correct answers
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    response_key = []
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
        # sort the contours or the current question from left to right, then initialize the index of the bubbled answer
        cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
        bubbled = None
        total = []
        # loop over the sorted contours
        for (j, c) in enumerate(cnts):
            # construct a mask that reveals only the current "bubble" for the question
            mask = np.zeros(warped_sub.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
            mask = cv2.bitwise_and(warped_sub, warped_sub, mask=mask)
            total.append((cv2.countNonZero(mask), j))

        norm_list = [float(i)/sum([i[0] for i in total]) for i in [i[0] for i in total]]
        if max(norm_list) > 0.3:
            bubbled = total[norm_list.index(max(norm_list))]
        else:
            bubbled = (None, None)
        response_key.append({b_no*30+sb_no*5+q+1: bubbled[1]})
        cv2.waitKey(0)
    return response_key


def processOMR(imagePath, blockNumber=1, answerKey=True):
    if answerKey:
        warped, paper = getContoursFromImage(imagePath=imagePath, blockNumber=blockNumber)
        finalList = []
        for b_no in range(0,5,1):
            for sb_no in range(0,6,1):
                warped1 = findContoursFromQASection(matrix=warped, b_no=b_no)
                warped2 = findContoursFromQAColumn(matrix=warped1, sb_no=sb_no)
                finalList = finalList + findMarkedCircles(warped_sub=warped2, b_no=b_no, sb_no=sb_no)
        return finalList
    else:
        return None


if __name__ == '__main__':
    start = datetime.now()
    blockNumber = 1
    imagePath = 'C:/Users/Swapnanil/Machine Learning/PwC/swap100.jpg' #200
    resp = processOMR(imagePath=imagePath, blockNumber=blockNumber, answerKey=True)
    print(resp)
    print("Total running time: %s" % (datetime.now() - start))