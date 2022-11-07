import cv2
import numpy as np
import toolbox as tb


"""
没有使用STC编码嵌入， 全部嵌入1
"""


def embed():
    gray = cv2.imread('lena.png', 0)
    # gray = gray[104:112, 320:328]
    table = tb.jpeg_quantization_table()
    jpg_dct = tb.jpeg_dct(gray)
    for u in range(0, gray.shape[0], 8):
        for v in range(0, gray.shape[1], 8):
            block = jpg_dct[u:u+8, v:v+8]
            for F in range(Fn, Fn+1):
                for i in range(F):
                    Delta = table[i, F-1-i]
                    # f -- positive
                    f = block[i, F-1-i]
                    absf = abs(f)
                    if absf % (2*Delta) == 0:
                        absf += 1
                    p = 0.17
                    if absf // Delta % 2 == 0 and absf % Delta < Delta / (2-p):
                        absf = absf // Delta * Delta + 2 * Delta/3
                    elif absf // Delta % 2 == 1 and absf % Delta > Delta / (2+p):
                        absf = absf // Delta * Delta + Delta/3
                    if f < 0:
                        block[i, F-1-i] = -absf
                    else:
                        block[i, F-1-i] = absf
                    # block[i, F-1-i] = absf * np.sign(f)


            jpg_dct[u:u+8, v:v+8] = block

    img = tb.jpeg_idct(jpg_dct)
    # 存为jpg
    q = 100
    # cv2.imwrite('ste.png', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    cv2.imwrite('ste.png', img)


def extract():
    gray = cv2.imread('ste.png', 0)
    table = tb.jpeg_quantization_table()
    jpg_dct = tb.jpeg_dct(gray)
    m = ''
    for u in range(0, gray.shape[0], 8):
        for v in range(0, gray.shape[1], 8):
            block = jpg_dct[u:u+8, v:v+8]
            for F in range(Fn, Fn+1):
                for i in range(F):
                    Delta = table[i, F-1-i]
                    # f -- positive
                    f = block[i, F-1-i]
                    if np.round(f / Delta) % 2 == 0:
                        m += '0'
                        print(u, v)
                    else:
                        m += '1'
    print(m)
    print(m.count('0')/len(m)*100, '%')


Fn = 5
embed()
extract()
