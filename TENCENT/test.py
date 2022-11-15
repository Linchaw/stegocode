import cv2
import numpy as np
import toolbox as tb

# IMG = cv2.imread('../lena.png')
# Y, U, V = cv2.split(cv2.cvtColor(IMG, cv2.COLOR_BGR2YUV))
#
# Ydct = tb.jpeg_dct(Y)
# table = tb.jpeg_quantization_table()
# print(table)
#
# for i in range(0, Ydct.shape[0], 8):
#     for j in range(0, Ydct.shape[1], 8):
#         Ydct[i+2, j+2] = table[2, 2]
#
# # print(np.round(Ydct[0:8, 0:8],1))
# Yidct = tb.jpeg_idct(Ydct)
# tb.imshow(Yidct)

# print(IMG.shape)

print((220+269+197+209+179+359+599)*0.7)
