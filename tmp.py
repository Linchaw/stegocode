
import cv2
import numpy as np
import toolbox as tb

Length = 2
F = 5

img = cv2.imread('lena.png', 0)
orb = cv2.ORB_create()
kp = orb.detect(img, None)
kp, des = orb.compute(img, kp)
for i in des:
    print(i)
