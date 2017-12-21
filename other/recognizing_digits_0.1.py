import matplotlib.pyplot as plt
import cv2

im = cv2.imread('datasets/1.png')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im2, contours, -1, (0, 255, 0), 3)

plt.imshow(im2)
plt.show()


