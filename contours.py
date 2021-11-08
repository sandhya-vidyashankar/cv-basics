import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/Sandhya/Documents/Python/OpenCV basics/shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[4]
img = cv2.drawContours(gray, [cnt], 0, (0, 255, 0), 3)

cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/dummy.jpg',img)

