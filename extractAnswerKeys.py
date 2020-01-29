# import the necessary packages
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from imutils.perspective import four_point_transform
import warnings
import math
warnings.filterwarnings("ignore")


def getContoursFromImage(imagePath, block_number):
    # load the image and compute the ratio of the old height to the new height, clone it, and resize it
    image = cv2.imread(imagePath)
    orig = image.copy()

    # convert the image to grayscale, blur it, and find edges in the image
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0, sigmaY=0)
    edged = cv2.Canny(image=gray, threshold1=75, threshold2=200, L2gradient=False)
    #cv2.imshow("gray", imutils.resize(gray, height=800))
    #cv2.imshow("edged", imutils.resize(edged, height=800))

    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(image=edged.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) #return countour tuple
    cnts = imutils.grab_contours(cnts) # extract cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:200]

    screenCnt = []
    total_cnts = len(cnts)
    print(f'Total contours = {total_cnts}')
    # loop over the contours
    for i in range(total_cnts):
        # approximate the contour
        peri = cv2.arcLength(cnts[i], True)
        approx = cv2.approxPolyDP(cnts[i], 0.02 * peri, True)
        # if our approximated contour has four points, then we can assume that we have found our screen
        if len(screenCnt)>100:
            break
        elif len(approx) == 4:
            if i == 0:
                screenCnt.append(approx)
            elif abs(cv2.contourArea(cnts[i])-cv2.contourArea(cnts[i-1])) > cv2.contourArea(cnts[i])*0.1:
                screenCnt.append(approx)
            else:
                pass
                #print(f'{cv2.contourArea(cnts[i])} is the area of {i} and {i-1} contour')

    total = [cv2.contourArea(cnt) for cnt in screenCnt]
    norm_list = [float(i)/sum([i for i in total]) for i in [i for i in total]]
    print(norm_list)
    for idx, norm in enumerate(norm_list):
        if idx==0:
            pass
            #print(f'++++++++{idx}')
        elif norm > max(norm_list)*0.90:
            #print(f'++++++++{idx}')
            del screenCnt[idx]

    print(f'Total unique contours = {len(screenCnt)}')

    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(image=gray, pts=screenCnt[block_number].reshape(4, 2))
    warped = cv2.threshold(src=warped, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    paper = four_point_transform(image=orig, pts=screenCnt[block_number].reshape(4, 2))
    # show the original and scanned images
    #cv2.drawContours(image=image, contours=[screenCnt[block_number]], contourIdx=-1, color=(0, 0, 255), thickness=15)
    #cv2.imshow(winname="Original", mat=imutils.resize(image, height=800))
    #cv2.imshow(winname="Scanned", mat=imutils.resize(warped, height=800))
    #cv2.waitKey(0)
    print(cv2.contourArea(screenCnt[block_number]))
    return warped, paper

def findMarkedCircles(warped_sub):
    # find contours in the thresholded image, then initialize the list of contours that correspond to questions
    cnts = cv2.findContours(warped_sub.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    print(f'------>>>>>Totl contours inside the image:{len(cnts)}')

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        # in order to label the contour as a question, region should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 30 and h >= 30 and ar >= 0.90 and ar <= 1.2:
            questionCnts.append(c)
    print(f'------>>>>>Total valid question contours:{len(questionCnts)}')

def findContoursFromQASection(matrix, b_no):
    # find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
    cnts = cv2.findContours(image=matrix.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) #return countour tuple
    cnts = imutils.grab_contours(cnts) # extract cnts[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    screenCnt=[]
    print(f'Total contours = {len(cnts)}')
    
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
            #print(f'++++++++{idx}')
        elif norm > max(norm_list)*0.90:
            #print(f'++++++++{idx}')
            del screenCnt[idx]
    
    # apply the four point transform to obtain a top-down view of the original image
    warped = four_point_transform(image=matrix, pts=screenCnt[b_no].reshape(4, 2))
    #warped = cv2.threshold(src=warped, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # show the original and scanned images
    cv2.drawContours(image=matrix, contours=[screenCnt[b_no]], contourIdx=-1, color=(0, 0, 255), thickness=15)
    cv2.imshow(winname="Original sub", mat=imutils.resize(matrix, height=800))
    cv2.imshow(winname="Scanned sub", mat=imutils.resize(warped, height=800))
    cv2.waitKey(0)
    print(f'Total valid contours:{len(screenCnt)}')

    for idx, cnt in enumerate(screenCnt):
        pass
        #print(f'{idx+1}th-----------{cv2.contourArea(cnts[idx])}----{cv2.contourArea(cnt)}')
    #print(cv2.arcLength(screenCnt[b_no], True))
    x_cor_for_max_circle = [cor[0] for cor in screenCnt[0].reshape(4,2)]
    x_cor_for_nth_circle = [cor[0] for cor in screenCnt[b_no].reshape(4,2)]
    print(f'Block Number: {math.ceil(min(x_cor_for_nth_circle)*5/max(x_cor_for_max_circle))}')

    print(f'Total unique contours = {len(screenCnt)}')
    return warped

def findContoursFromQAColumn(matrix, sb_no):
    sub = matrix[(sb_no)*int(matrix.shape[0]/6):(sb_no+1)*int(matrix.shape[0]/6),0:matrix.shape[1]]
    cv2.imshow(winname="Original subsub", mat=imutils.resize(sub, height=200))
    cv2.waitKey(0)

block_number = 1#fixed
b_no = 3
sb_no = 0
imagePath = 'C:/Users/Swapnanil/Machine Learning/PwC/muthu1.jpg'
warped, paper=getContoursFromImage(imagePath=imagePath, block_number=block_number)
#print(warped)
findMarkedCircles(warped_sub=warped)
for b_no in [3,1,2,4,5]:
    warped1 = findContoursFromQASection(matrix=warped, b_no=b_no)
    findContoursFromQAColumn(matrix=warped1, sb_no=sb_no)
    break

