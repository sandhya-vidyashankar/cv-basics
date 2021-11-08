import numpy as np
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('/Users/Sandhya/Documents/Python/OpenCV basics/rome.jpg', 0)
image_enhanced = cv2.equalizeHist(img)
edges = cv2.Canny(image_enhanced,100,200)

cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/dummy.jpg',img)
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/dummy2.jpg',image_enhanced)
cv2.imwrite('/Users/Sandhya/Documents/Python/OpenCV basics/Results/dummy3.jpg',edges)