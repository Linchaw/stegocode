import cv2
import toolbox as tb
import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(matrix)
dct = cv2.dct(np.float32(matrix))
print(dct)


