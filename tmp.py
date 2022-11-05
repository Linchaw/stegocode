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

F = 5


def find_key_axis(gray: object, text=800) -> object:
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

    return key_axis


def embed():
    img = cv2.imread('lena.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key_axis = find_key_axis(gray)
    key_axis.sort(key=lambda x: x[1])
    key_axis.sort(key=lambda x: x[0])
    # print(key_axis)
    b, g, r = cv2.split(img)
    table = tb.jpeg_quantization_table()

    for row, column in key_axis:
        bb = b[row - 4:row + 4, column - 4:column + 4]
        bg = g[row - 4:row + 4, column - 4:column + 4]
        qdbb = cv2.dct(np.float32(bb)) / table
        qdbg = cv2.dct(np.float32(bg)) / table

        cB, cG, flag = (0, 0, False)
        for i in range(F):
            cB += qdbb[i, F-i]
            cG += qdbg[i, F-i]

        if cB <= cG:
            flag = True

        if flag:
            for i in range(F):
                qdbb[i, F-i] = qdbg[i, F-i] + 1
            bb = cv2.idct(qdbb * table)
            b[row - 4:row + 4, column - 4:column + 4] = np.uint8(bb)

    img = cv2.merge([b, g, r])
    cv2.imwrite('ste.png', img)


def extract():
    img = cv2.imread('ste.png')
    # resize 1.5
    f = 0.8
    img = cv2.resize(img, (0, 0), fx=f, fy=f)
    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    key_axis = find_key_axis(gray)
    key_axis.sort(key=lambda x: x[1])
    key_axis.sort(key=lambda x: x[0])
    # print(key_axis)
    b, g, r = cv2.split(img)
    table = tb.jpeg_quantization_table()

    mes = ""
    for row, column in key_axis:
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
    print(mes.count('1')/len(mes)*100, '%')


embed()
extract()
