import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/Sandhya/Documents/Python/OpenCV basics/coffee.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/1grayscale.jpg',gray)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#remove noise
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,kernel, iterations = 2)
sure_bg = cv2.dilate(opening,kernel,iterations=3)
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/2surebg.jpg',sure_bg)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/3surefg.jpg',sure_fg)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/4unknown.jpg',unknown)

# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv2.watershed(img,markers)
img[markers == -1] = [255,0,0]
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/5marked.jpg',markers)
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/6result.jpg',img)