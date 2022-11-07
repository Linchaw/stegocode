
import cv2
import numpy as np
import toolbox as tb
import pywt
F = 5


def mes_bkps(gray, kp_list, threshold=350):
    kpxy = []
    qtable = tb.jpeg_quantization_table()
    for skp in kp_list:
        xf, yf = skp.pt
        y, x = int(xf), int(yf)
        block = []
        dct_block = []
        qdct_block = []
        block.append(gray[x-4:x, y-4:y])
        block.append(gray[x-4:x, y:y+4])
        block.append(gray[x:x+4, y-4:y])
        block.append(gray[x:x+4, y:y+4])

        for i in range(4):
            dct_block.append(cv2.dct(np.float32(block[i])))
            qdct_block.append(np.int16(dct_block[i] / qtable[:4, :4]))
        rf, ef = (0, 0)
        for i in range(4):
            rf += np.sum((qdct_block[i] != 0).astype(np.int_))
            ef += np.sum(np.abs(qdct_block[i])) - np.abs(qdct_block[i][0, 0])
        si = rf + ef + ef * rf
        if si > threshold:
            kpxy.append((x, y))

    return kpxy


def mes_bkps8(gray, kp_list, threshold=200):
    qtable = tb.jpeg_quantization_table()
    kpxy = []
    for skp in kp_list:
        xf, yf = skp.pt
        x, y = int(xf), int(yf)
        block = gray[y-4:y+4, x-4:x+4]
        dct_block = cv2.dct(np.float32(block))
        qdct_block = np.int16(dct_block / qtable)
        rfb = (qdct_block != 0).astype(np.int_)
        rf = np.sum(rfb) - 1
        ef = np.sum(np.abs(qdct_block)) - qdct_block[0, 0]
        si = rf + ef + ef * rf
        if si > threshold:
            kpxy.append((y, x))

    return kpxy


def modify(b, g, kpxy, mes, f=F):
    mes_str = tb.bytes2binstr(mes)
    print("嵌入信息：", mes_str)
    table = tb.jpeg_quantization_table()
    idx = 0
    for x, y in kpxy:
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        dct_b_block = cv2.dct(np.float32(b_block)) / table
        dct_g_block = cv2.dct(np.float32(g_block)) / table

        cB, cG, flag = (0, 0, False)
        for i in range(f):
            cB += dct_b_block[i, f - i]
            cG += dct_g_block[i, f - i]
        # print(cB, cG)
        sub_mes = mes_str[idx]

        if sub_mes == '1' and cB <= cG:
            flag = 1
            for i in range(f):
                dct_b_block[i, f - i] = dct_g_block[i, f - i] + 2

        elif sub_mes == '0' and cB > cG:
            flag = 1
            for i in range(f):
                dct_b_block[i, f - i] = dct_g_block[i, f - i] - 2

        if flag:
            block = cv2.idct(dct_b_block * table)
            block = np.uint8(block)
            b[x - 4:x + 4, y - 4:y + 4] = block

        idx += 1

        if idx == len(mes_str):
            break

    if idx < len(mes_str) - 1:
        print("未嵌入所有数据")

    return b


def modify2(b, g, kpxy, mes, f=3):
    mes_str = tb.bytes2binstr(mes)
    print("嵌入信息：", mes_str)
    idx = 0
    for x, y in kpxy:
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        # dwt2
        b_dwt = pywt.dwt2(b_block, 'db1')
        g_dwt = pywt.dwt2(g_block, 'db1')

        b_dwt_dct = cv2.dct(np.float32(b_dwt[0]))
        g_dwt_dct = cv2.dct(np.float32(g_dwt[0]))

        cB, cG, flag = (0, 0, False)
        for i in range(f):
            cB += b_dwt_dct[i, f - i]
            cG += g_dwt_dct[i, f - i]
        # print(cB, cG)
        sub_mes = mes_str[idx]

        if sub_mes == '1' and cB <= cG:
            flag = 1
            for i in range(f):
                b_dwt_dct[i, f - i] = g_dwt_dct[i, f - i] + 1

        elif sub_mes == '0' and cB > cG:
            flag = 1
            for i in range(f):
                b_dwt_dct[i, f - i] = g_dwt_dct[i, f - i] - 2

        if flag:
            block = cv2.idct(b_dwt_dct)
            block = np.float64(block)
            block = pywt.idwt2((block, b_dwt[1]), 'db1')
            b[x - 4:x + 4, y - 4:y + 4] = block

        idx += 1

        if idx == len(mes_str):
            break

    if idx < len(mes_str) - 1:
        print("未嵌入所有数据")

    return b


def extract_mes(b, g, kpxy, L):
    mes_str = ''
    for x, y in kpxy:
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        dct_b_block = cv2.dct(np.float32(b_block))
        dct_g_block = cv2.dct(np.float32(g_block))
        cB, cG = (0, 0)
        for i in range(F):
            cB += dct_b_block[i, F - i]
            cG += dct_g_block[i, F - i]
        if cB > cG:
            mes_str += '1'
        else:
            mes_str += '0'
        if len(mes_str) == L * 8:
            break
    return mes_str


def extract_mes2(b, g, kpxy, L, f=3):
    mes_str = ''
    for x, y in kpxy:
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        # dwt2
        b_dwt = pywt.dwt2(b_block, 'db1')
        g_dwt = pywt.dwt2(g_block, 'db1')

        b_dwt_dct = cv2.dct(np.float32(b_dwt[0]))
        g_dwt_dct = cv2.dct(np.float32(g_dwt[0]))

        cB, cG = (0, 0)
        for i in range(f):
            cB += b_dwt_dct[i, f - i]
            cG += g_dwt_dct[i, f - i]
        if cB > cG:
            mes_str += '1'
        else:
            mes_str += '0'
        if len(mes_str) == L * 8:
            break
    return mes_str


def embed(img, mes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    kp_list = list(kp)
    for skp in kp:
        x, y = skp.pt
        for sskp in kp_list:
            sx, sy = sskp.pt
            if (x - sx) ** 2 + (y - sy) ** 2 <= 32 and sskp.response < skp.response:
                kp_list.remove(sskp)

    kpxy = mes_bkps8(gray, kp_list)
    kpxy.sort(key=lambda x: x[1])
    kpxy.sort(key=lambda x: x[0])

    print("嵌入点：", kpxy)
    # 嵌入消息
    b, g, r = cv2.split(img)
    b = modify(b, g, kpxy, mes)
    ste_img = cv2.merge([b, g, r])
    # 保存嵌入后的图像 质量为95
    cv2.imwrite('ste.png', ste_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def extract(ste_img, L):
    gray = cv2.cvtColor(ste_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)
    kp_list = list(kp)
    for skp in kp:
        x, y = skp.pt
        for sskp in kp_list:
            sx, sy = sskp.pt
            if (x - sx) ** 2 + (y - sy) ** 2 <= 32 and sskp.response < skp.response:
                kp_list.remove(sskp)

    kpxy = mes_bkps8(gray, kp_list)
    kpxy.sort(key=lambda x: x[1])
    kpxy.sort(key=lambda x: x[0])
    print("提取点：", kpxy)

    b, g, r = cv2.split(ste_img)
    mes_str = extract_mes(b, g, kpxy, L)

    print("提取信息：", mes_str)


def main():
    # 嵌入消息
    em_flag = 1
    L = 2
    if em_flag:
        img = cv2.imread('lena.png')
        mes = tb.new_rand_bytes(L, 2)
        embed(img, mes)

    # 提取消息
    ex_flag = 1
    if ex_flag:
        filename = '../ste.png'
        ste_img = cv2.imread(filename)
        ste_img = cv2.resize(ste_img, (512, 512))
        extract(ste_img, L)

    pass


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
