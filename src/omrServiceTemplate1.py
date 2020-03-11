import logging as logger
import numpy as np
import cv2
import imutils
from imutils.perspective import four_point_transform
import warnings
import math
from imutils import contours
import sys
import names
warnings.filterwarnings("ignore")

from .cornerFeatures import cornerFeatures
from .commonConstants import *

class omrServiceTemplate1:

    def getGrayImage(self, imagePath):
        print(imagePath)
        # load the image and compute the ratio of the old height to the new height, clone it, and resize it
        image = cv2.imread(imagePath)
        orig = image.copy()
        # convert the image to grayscale, blur it, and find edges in the image
        gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(src=gray, ksize=KSIZE_GAUBLUR, sigmaX=SIGMAX_GAUBLUR, sigmaY=SIGMAY_GAUBLUR)
        return gray
    
    def cannyFilter(self, grayImage):
        return cv2.Canny(image=grayImage, threshold1=THRESHOLD1_CANNY, threshold2=THRESHOLD2_CANNY,
                         L2gradient=True, apertureSize=APERTURESIZE_CANNY)
    
    def getDistinctContours(self, imageMatrix, maxContours=2):
        cnts = cv2.findContours(image=imageMatrix.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        screenCnt = []
        # loop over the contours
        for idx, cnt in enumerate(cnts):
            # approximate the contour
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(curve=cnt, epsilon=0.02*peri, closed=True)
            # if our approximated contour has four points, then we can assume that we have found our screen
            if len(screenCnt) > maxContours*10:
                break
            elif len(approx) == 4:
                if idx == 0:
                    screenCnt.append(approx)
                else:
                    screenCnt.append(approx)
        total = [cv2.contourArea(cnt) for cnt in screenCnt]
        norm_list = [float(i)/sum([i for i in total]) for i in [i for i in total]]
        newScreenCnt = []
        for idx, norm in enumerate(norm_list):
            if idx == 0:
                newScreenCnt.append(screenCnt[idx])
            elif norm > max(norm_list)*0.90:
                pass
            else:
                newScreenCnt.append(screenCnt[idx])
        return newScreenCnt[:maxContours]
    
    def getNContours(self, grayImage, maxContours):
        edged = self.cannyFilter(grayImage=grayImage)
        # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
        screenCnt = self.getDistinctContours(imageMatrix=edged, maxContours=maxContours)
        return screenCnt
    
    def findOuterBoundary(self, grayImage):
        return self.getNContours(grayImage, maxContours=1)[0]
    
    
    def getQASectionFromImage(self, grayImage):
        # apply the four point transform to obtain a top-down view of the original image
        warped = four_point_transform(image=grayImage, pts=self.getNContours(grayImage, maxContours=2)[1].reshape(4, 2))
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return warped
    
    def getFourPointTransform(self, image, pts):
        return four_point_transform(image=image, pts=pts.reshape(4, 2))
    
    def findContoursFromQASection(self, matrix, b_no):
        screenCnt = self.getDistinctContours(imageMatrix=matrix, maxContours=6)
        x_cor_for_max_circle = [cor[0] for cor in screenCnt[0].reshape(4, 2)]
        for contr in screenCnt[1:6]:
            x_cor_for_nth_circle = [cor[0] for cor in contr.reshape(4, 2)]
            if max(x_cor_for_nth_circle) <= math.ceil((b_no+1)*max(x_cor_for_max_circle)/5) and \
               min(x_cor_for_nth_circle) >= math.ceil(b_no*max(x_cor_for_max_circle)/5):
                return contr
            else:
                pass
    
    def findMarkedCircles(self, warped, contr, b_no):
        # apply the four point transform to obtain a top-down view of the original image
        warped = self.getFourPointTransform(image=warped, pts=contr)
        warped = warped[:, int(0.215*warped.shape[1]):]
        # find contours in the thresholded image, then initialize the list of contours that correspond to questions
        #warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        cnts = cv2.findContours(warped.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        print(f'------>>>>>Total contours inside the image:{len(cnts)}')
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
    
            # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
            # have an aspect ratio approximately equal to 1
            if MAX_WIDTH >= w >= MIN_WIDTH and MAX_HEIGHT >= h >= MIN_HEIGHT and ar >= MIN_ASP_RATIO_CONTR and ar <= MAX_ASP_RATIO_CONTR:
                questionCnts.append(c)
    
        print(f'------>>>>>Total valid question contours:{len(questionCnts)}')
        # sort the question cofntours top-to-bottom, then initialize the total number of correct answers
        questionCnts = contours.sort_contours(questionCnts[:120], method="top-to-bottom")[0]
        response_key = []
    
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
            # sort the contours or the current question from left to right, then initialize the index of the bubbled answer
            cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
            bubbled = NONE
            total = []
            # loop over the sorted contours
            for (j, c) in enumerate(cnts):
                # construct a mask that reveals only the current "bubble" for the question
                mask = np.zeros(warped.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
                # apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
                mask = cv2.bitwise_and(warped, warped, mask=mask)
                total.append((cv2.countNonZero(mask), j))
            norm_list1 = [float(i)/sum([i[0] for i in total]) for i in [i[0] for i in total]]
            print(f'{q+1}-------{[i[0] for i in total]}----{norm_list1}')
            sort_norm_list1 = sorted(norm_list1)
    
            if sort_norm_list1[0] <= THRS_2_MARKED_AMONG_4:
                if sort_norm_list1[1] <= THRS_2_MARKED_AMONG_4:
                    bubbled = (NONE, -1)
                elif sort_norm_list1[-1] >= THRS_3_MARKED_AMONG_4:
                    bubbled = (NONE, -1)
                else:
                    bubbled = total[norm_list1.index(min(norm_list1))]
            else:
                bubbled = (NONE, NONE)
    
            if bubbled[1] is NONE:
                resp = ""
            elif bubbled[1] == 0:
                resp = "A"
            elif bubbled[1] == 1:
                resp = "B"
            elif bubbled[1] == 2:
                resp = "C"
            elif bubbled[1] == 3:
                resp = "D"
            elif bubbled[1] == -1:
                resp = "Z"
    
            response_key.append({"questionno": str(b_no*NO_OF_QUES_PER_BLOCK+q+1), "answer": resp})
        return response_key
    
    def concatenate_list_data(self, list):
        result = ''
        for element in list:
            result += str(element)
        return result
    
    def findSchoolCode(self, grayImage, totalRowColumn=100):
        warped = self.getFourPointTransform(image=grayImage, pts=self.findOuterBoundary(grayImage))
        getCorners = cornerFeatures.getSchoolCodeCorners()
        x1, x2 = (int(getCorners[0]*warped.shape[0])), (int(getCorners[1]*warped.shape[0]))
        y1, y2 = (int(getCorners[2]*warped.shape[1])), (int(getCorners[3]*warped.shape[1]))
        warped = warped[x1:x2, y1:y2]
        return self.findInformationBlock(warped, totalRowColumn)
    
    def findRollNumber(self, grayImage, totalRowColumn=100):
        warped = self.getFourPointTransform(image=grayImage, pts=self.findOuterBoundary(grayImage))
        getCorners = cornerFeatures.getRollNumberCorners()
        x1, x2 = (int(getCorners[0]*warped.shape[0])), (int(getCorners[1]*warped.shape[0]))
        y1, y2 = (int(getCorners[2]*warped.shape[1])), (int(getCorners[3]*warped.shape[1]))
        warped = warped[x1:x2, y1:y2]
        return self.findInformationBlock(warped, totalRowColumn)
    
    def findClass(self, grayImage, totalRowColumn=20):
        warped = self.getFourPointTransform(image=grayImage, pts=self.findOuterBoundary(grayImage))
        getCorners = cornerFeatures.getClassCorners()
        x1, x2 = (int(getCorners[0]*warped.shape[0])), (int(getCorners[1]*warped.shape[0]))
        y1, y2 = (int(getCorners[2]*warped.shape[1])), (int(getCorners[3]*warped.shape[1]))
        warped = warped[x1:x2, y1:y2]
        return self.findInformationBlock(warped, totalRowColumn)
    
    def findSection(self, grayImage, totalRowColumn=10):
        warped = self.getFourPointTransform(image=grayImage, pts=self.findOuterBoundary(grayImage))
        getCorners = cornerFeatures.getSectionCorners()
        x1, x2 = (int(getCorners[0]*warped.shape[0])), (int(getCorners[1]*warped.shape[0]))
        y1, y2 = (int(getCorners[2]*warped.shape[1])), (int(getCorners[3]*warped.shape[1]))
        warped = warped[x1:x2, y1:y2]
        return self.findInformationBlock(warped, totalRowColumn)
    
    
    def findInformationBlock(self, warped, totalRowColumn):
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
        warped = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
        cnts = cv2.findContours(warped.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
        cnts = imutils.grab_contours(cnts)
        print(f'Total contours in the Block: {len(cnts)}')
        questionCnts = []
        for c in cnts:
            # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
            # have an aspect ratio approximately equal to 1
            if MAX_WIDTH >= w >= MIN_WIDTH and MAX_HEIGHT >= h >= MIN_HEIGHT and ar >= MIN_ASP_RATIO_CONTR and ar <= MAX_ASP_RATIO_CONTR:
                questionCnts.append(c)
        print(f'Total relevance contours in the Block: {len(questionCnts)}', file=sys.stderr)
        questionCnts = contours.sort_contours(questionCnts[:totalRowColumn], method=TOP_TO_BOTTOM)[0]
        response_key = []
    
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 10)):
            cnts = contours.sort_contours(questionCnts[i:i + 10])[0]
            bubbled = NONE
            total = []
            # loop over the sorted contours
            for (j, c) in enumerate(cnts):
                mask = np.zeros(warped.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, (255, 0, 0), -1)
                # apply the mask to the thresholded image, then count the number of non-zero pixels in the bubble area
                mask = cv2.bitwise_and(warped, warped, mask=mask)
                total.append((cv2.countNonZero(mask), j))
    
            norm_list1 = [float(i)/sum([i[0] for i in total]) for i in [i[0] for i in total]]
            print(f'Block {q}: {[i[0] for i in total]}-----------{norm_list1}', file=sys.stderr)
            if min(norm_list1) <= THRS_1_MARKED_AMONG_10:
                bubbled = total[norm_list1.index(min(norm_list1))]
                response_key.append({str(q): (9 - bubbled[1])})
            else:
                bubbled = (NONE, -1)
    
        if -1 in [list(i.values())[0] for i in response_key]:
            return BLOCK_PARSING_ERR_MSG, 500
        else:
            return self.concatenate_list_data([list(i.values())[0] for i in response_key])
    
    def processOMR(self, imagePath, answerKey=True, noOfQuestions=MAX_QUES_TEMP1):
        if answerKey == TRUE:
            gray = self.getGrayImage(imagePath=imagePath)
            warped = self.getQASectionFromImage(grayImage=gray)
            finalList = []
            for b_no in range(0, 5, 1):
                contr = self.findContoursFromQASection(matrix=warped, b_no=b_no)
                finalList = finalList + self.findMarkedCircles(warped=warped, contr=contr, b_no=b_no)
            finalList = finalList[0:noOfQuestions]
            error_index = []
            for i in range(int(noOfQuestions)):
                if list(finalList[i].values())[0] == EMPTY:
                    error_index.append(i+1)
            for i in range(int(noOfQuestions)):
                if len(error_index) != 0:
                    return NONE, FAILED, error_index
            if len([i['answer'] for i in finalList if i['answer'] == EMPTY]) == 0:
                return finalList, OK, NONE
            else:
                return NONE, FAILED, error_index
        elif answerKey == FALSE:
            gray = self.getGrayImage(imagePath=imagePath)
            warped = self.getQASectionFromImage(grayImage=gray)
            finalList = []
            for b_no in range(0, 5, 1):
                contr = self.findContoursFromQASection(matrix=warped, b_no=b_no)
                finalList = finalList + self.findMarkedCircles(warped=warped, contr=contr, b_no=b_no)
            finalList = finalList[0:noOfQuestions]
            print(f'Answer List:-----------{finalList}', file=sys.stderr)
            return finalList, OK, NONE
    
    def otherDetailsOfStudent(self, imagePath):
        gray = self.getGrayImage(imagePath=imagePath)
        rollNumber = self.findRollNumber(grayImage=gray)
        schoolCode = self.findSchoolCode(grayImage=gray)
        classs = self.findClass(grayImage=gray)
        section = self.findSection(grayImage=gray)
        studentName = names.get_full_name()
        print(f'----->>>>Roll no is....................:{rollNumber}', file=sys.stderr)
        print(f'----->>>>School Code no is.............:{schoolCode}', file=sys.stderr)
        print(f'----->>>>Class no is...................:{classs}', file=sys.stderr)
        print(f'----->>>>Section no is.................:{section}', file=sys.stderr)
        return rollNumber, schoolCode, studentName, classs, section