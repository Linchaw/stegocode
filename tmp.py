"""
kp.pt 为关键点的坐标 为一个元组 为（column,row）
img.shape 为图像的大小 为一个元组 为（row,column,channel）
Specific methods:
1. 通过ORB检测嵌入点
    1. 通过ORB检测关键点
    2. 通过ORB检测关键点的坐标
    3. 以关键点的坐标为中心，以4为半径，截取一个8*8的区域
    4. 对截取的区域进行DCT变换
    5. 对DCT变换后的区域进行量化
    6. 计算量化后的区域的纹理特征
    7. 通过纹理特征判断是否为嵌入点
    8. 将嵌入点的坐标存入key_axis中
2. 在嵌入点嵌入信息

"""
import cv2
import toolbox as tb
import numpy as np

F = 6
L = 5
p = 0.5
m = tb.new_rand_bytes(L)
m = tb.bytes2binstr(m)
print(m)


def find_key_axis(gray: object, text=1200) -> object:
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    kp_list = list(kp)
    for skp in kp:
        x, y = skp.pt
        for sskp in kp_list:
            sx, sy = sskp.pt
            if (x - sx) ** 2 + (y - sy) ** 2 <= 32 and sskp.response < skp.response:
                kp_list.remove(sskp)

    qtable = tb.jpeg_quantization_table()
    key_axis = []
    for skp in kp_list:
        column, row = skp.pt
        column = int(column)
        row = int(row)
        block = gray[row - 4:row + 4, column - 4:column + 4]
        dct_block = cv2.dct(np.float32(block))
        qdct_block = np.round(dct_block / qtable)
        # print(qdct_block)
        rfb = (qdct_block != 0).astype(np.int_)
        rf = np.sum(rfb) - 1
        ef = np.sum(np.abs(qdct_block)) - qdct_block[0, 0]
        si = rf + ef + ef * rf
        if si > text:
            key_axis.append((row, column))

    key_axis.sort(key=lambda x: x[1])
    key_axis.sort(key=lambda x: x[0])
    return key_axis


def embed():
    img = cv2.imread('lena.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key_axis = find_key_axis(gray)
    # print(key_axis)
    b, g, r = cv2.split(img)
    table = tb.jpeg_quantization_table()
    idx = 0

    print(key_axis[:L*8])
    for row, column in key_axis[:L*8]:
        bb = b[row - 4:row + 4, column - 4:column + 4]
        bg = g[row - 4:row + 4, column - 4:column + 4]
        dbb = cv2.dct(np.float32(bb))
        qdbb = np.round(dbb / table)
        dbg = cv2.dct(np.float32(bg))
        qdbg = np.round(dbg / table)

        cB, cG, flag = (0, 0, False)
        for i in range(F):
            cB += qdbb[i, F-i]
            cG += qdbg[i, F-i]

        if m[idx] == '1':
            if cB - cG <= 0:
                flag = True
                for i in range(F):
                    qdbb[i, F-i] = qdbg[i, F-i] + p

        else:
            if cB - cG >= 0:
                flag = True
                for i in range(F):
                    qdbb[i, F-i] = qdbg[i, F-i] - p

        if flag:
            bb = cv2.idct(qdbb * table)
            b[row - 4:row + 4, column - 4:column + 4] = np.uint8(bb)

        idx += 1

    img = cv2.merge([b, g, r])
    cv2.imwrite('ste.png', img)


def extract():
    img = cv2.imread('ste.png')
    # resize 1.5
    # f = 1.5
    # img = cv2.resize(img, (0, 0), fx=f, fy=f)
    # img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key_axis = find_key_axis(gray)

    # print(key_axis)
    b, g, r = cv2.split(img)
    table = tb.jpeg_quantization_table()

    mes = ""
    print(key_axis[:L * 8])
    for row, column in key_axis[:L*8]:
        bb = b[row - 4:row + 4, column - 4:column + 4]
        bg = g[row - 4:row + 4, column - 4:column + 4]
        dbb = cv2.dct(np.float32(bb))
        dbg = cv2.dct(np.float32(bg))

        cB, cG = (0, 0)
        for i in range(F):
            cB += dbb[i, F-i]
            cG += dbg[i, F-i]

        if cB > cG:
            mes += "1"
        else:
            mes += "0"

    print(mes)
    print(tb.error_rate(m, mes)*100, "%")


embed()
extract()

#  crop 20 % of the image --- 100% EXTRACTED
#  resize 0.5 of the image --- 81.25% EXTRACTED
#  resize 1.5 of the image --- 77.58% EXTRACTED