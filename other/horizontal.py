import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("datasets/1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.bitwise_not(img)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
plt.imshow(th2)
plt.show()

horizontal = th2
vertical = th2
rows, cols = horizontal.shape
horizontalsize = cols / 15
horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
# inverse the image, so that lines are black for masking
horizontal_inv = cv2.bitwise_not(horizontal)
# perform bitwise_and to mask the lines with provided mask
masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
# reverse the image back to normal
masked_img_inv = cv2.bitwise_not(masked_img)
plt.imshow(masked_img_inv)
plt.show()
