import cv2
import toolbox as tb

gray = cv2.imread('lena_75.jpg', 0)
tb.imshow(gray, 'gray')
img = cv2.imread('lena_75.jpg')
Y, U, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))

print(Y.shape, U.shape, V.shape)
tb.imshow(Y, 'Y')
tb.imshow(U, 'U')
tb.imshow(V, 'V')

recover = cv2.cvtColor(cv2.merge([Y, U, V]), cv2.COLOR_YUV2BGR)
tb.imshow(recover, 'recover')

# for i in range(70, 80):
#     table = tb.jpeg_quantization_table(i)
#     print(table)
