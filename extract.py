import cv2
import numpy as np
import toolbox as tb


em_flag = 1
Length = 2
threshold = 600
F = 5

stego = cv2.imread('stego.png')
# stego = cv2.imread('lena.png')
gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
b, g, r = cv2.split(stego)

orb = cv2.ORB_create()
kp = orb.detect(gray, None)   # type - tuple
kp_list = list(kp)
for skp in kp:
    x, y = skp.pt
    for sskp in kp_list:
        sx, sy = sskp.pt
        if (x-sx)**2 + (y-sy)**2 <= 32 and sskp.response < skp.response:
            kp_list.remove(sskp)
kp = tuple(kp_list)

qtable = tb.jpeg_quantization_table(75)
bkp = []
for skp in kp:
    xf, yf = skp.pt
    x, y = int(xf), int(yf)
    block = gray[x-4:x+4, y-4:y+4]
    dct_block = cv2.dct(np.float32(block))
    qdct_block = np.int16(dct_block / qtable)
    rfb = (qdct_block != 0).astype(np.int_)
    rf = np.sum(rfb) - 1
    ef = np.sum(np.abs(qdct_block)) - qdct_block[0, 0]
    si = rf + ef + ef * rf
    if si > threshold:
        bkp.append(skp)

bkp.sort(key=lambda x: x.response, reverse=True)
kp = tuple(bkp[:Length*8])

mes = ''
idx = 0
for skp in kp:
    xf, yf = skp.pt
    x, y = int(xf), int(yf)
    b_block = b[x - 4:x + 4, y - 4:y + 4]
    g_block = g[x - 4:x + 4, y - 4:y + 4]
    dct_b_block = cv2.dct(np.float32(b_block))
    dct_g_block = cv2.dct(np.float32(g_block))
    f = F
    cB, cG, flag = 0, 0, 0
    for i in range(f):
        cB += dct_b_block[i, f - i]
        cG += dct_g_block[i, f - i]

    if cB > cG:
        mes += '1'
    else:
        mes += '0'
    idx += 1

print(mes)
