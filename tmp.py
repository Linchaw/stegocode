import cv2
import numpy as np
import toolbox as tb
import pywt
from matplotlib import pyplot as plt
import PIL

from pywt import dwt2, idwt2

img = cv2.imread('half_lena.png')
gray = cv2.imread('half_lena.png', 0)
orb = cv2.ORB_create()
kp = orb.detect(gray, None)

img2 = cv2.drawKeypoints(img, kp, None)

# x, y = kp[0].pt
# x, y = int(x),int(y)
# print(x,y)
# r= 10
# img3 = img[y-r:y+r,x-r:x+r]
tb.imshow(img2)
# tb.imshow(img3)
