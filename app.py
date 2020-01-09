# import the necessary packages
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
block_number = 7

# load the image and compute the ratio of the old height to the new height, clone it, and resize it
image = cv2.imread('C:/Users/Swapnanil/Machine Learning/PwC/muthu.jpg')
orig = image.copy()

# convert the image to grayscale, blur it, and find edges in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

# find the contours in the edged image, keeping only the largest ones, and initialize the screen contour
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

screenCnt = []
total_cnts = len(cnts)
# loop over the contours
#for c in cnts:
for i in range(total_cnts):
    # approximate the contour
    peri = cv2.arcLength(cnts[i], True)
    approx = cv2.approxPolyDP(cnts[i], 0.02 * peri, True)
 
    # if our approximated contour has four points, then we can assume that we have found our screen
    if len(approx) == 4 and i==0:
        screenCnt.append(approx)
    elif len(approx) == 4 and abs(cv2.contourArea(cnts[i])-cv2.contourArea(cnts[i-1])) > cv2.contourArea(cnts[i])*0.15:
         screenCnt.append(approx)

# apply the four point transform to obtain a top-down view of the original image
warped = four_point_transform(orig, screenCnt[block_number].reshape(4, 2))
#resize_image = four_point_transform(orig, screenCnt[0].reshape(4, 2))

# convert the warped image to grayscale, then threshold it to give it that 'black and white' paper effect
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(image=warped, 
                    block_size=11, 
                    offset=10, 
                    method="gaussian")
warped = (warped > T).astype("uint8") * 255
#resize_image = cv2.cvtColor(resize_image, cv2.COLOR_BGR2GRAY)

# show the original and scanned images
print("Apply perspective transform")
cv2.drawContours(image, [screenCnt[block_number]], -1, (0, 0, 255), 15)
cv2.imshow("Original", imutils.resize(image, height=800))
cv2.imshow("Scanned", imutils.resize(warped, height=800))
cv2.waitKey(0)
