
import cv2
import numpy as np
import toolbox as tb
F = 6


def feature_block(gray):
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)   # type - tuple

    kp_list = list(kp)
    for skp in kp:
        x, y = skp.pt
        for sskp in kp_list:
            sx, sy = sskp.pt
            if (x-sx)**2 + (y-sy)**2 <= 32 and sskp.response < skp.response:
                kp_list.remove(sskp)

    return tuple(kp_list)


def mes_block(gray, kp, threshold=1000):
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
    return bkp


def modify_m(mes, b, g, kpxy, f=F):
    mes_str = tb.bytes2binstr(mes)
    print("嵌入信息：", mes_str)
    idx = 0
    for x, y in kpxy:
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        dct_b_block = cv2.dct(np.float32(b_block))
        dct_g_block = cv2.dct(np.float32(g_block))
        cB, cG, flag = (0, 0, False)
        for i in range(f):
            cB += dct_b_block[i, f-i]
            cG += dct_g_block[i, f-i]
        # print(cB, cG)
        sub_mes = mes_str[idx]

        if sub_mes == '1' and cB <= cG:
            flag = 1
            for i in range(f):
                dct_b_block[i, f - i] = dct_g_block[i, f - i] + 1

        elif sub_mes == '0' and cB > cG:
            flag = 1
            for i in range(f):
                dct_b_block[i, f - i] = dct_g_block[i, f - i] - 1

        if flag:
            block = cv2.idct(dct_b_block)
            block = np.uint8(block)
            b[x - 4:x + 4, y - 4:y + 4] = block

        idx += 1

    if idx < len(mes_str) - 1:
        print("未嵌入所有数据")

    return b


def get_info(kpxy, b, g, length, f=F):
    info = ''
    for x, y in kpxy:
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        dct_b_block = cv2.dct(np.float32(b_block))
        dct_g_block = cv2.dct(np.float32(g_block))
        cB, cG = (0, 0)
        for i in range(f):
            cB += dct_b_block[i, f - i]
            cG += dct_g_block[i, f - i]

        if cB > cG:
            info += '1'
        else:
            info += '0'

        if len(info) >= length*8:
            break

    return info


def embed(img, mes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)
    # 获取嵌入块
    kp = feature_block(gray)
    bkp = mes_block(gray, kp, 300)

    bkp.sort(key=lambda x: x.response, reverse=True)
    kpxy = [(int(x.pt[0]), int(x.pt[1])) for x in bkp[:len(mes)*8]]

    # 嵌入
    b = modify_m(mes, b, g, kpxy, F)
    ste_img = cv2.merge([b, g, r])
    # print(kpxy)
    return ste_img


def extract(ste_img, length):
    gray = cv2.cvtColor(ste_img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(ste_img)
    # 获取嵌入块
    kp = feature_block(gray)
    bkp = mes_block(gray, kp, 300)

    bkp.sort(key=lambda x: x.response, reverse=True)
    kpxy = [(int(x.pt[0]), int(x.pt[1])) for x in bkp[:length * 8]]

    info = get_info(kpxy, b, g, length, F)
    # print(kpxy)
    return info


def crop(img, radio):
    h, w = img.shape[:2]
    x = int(h*radio//4)
    y = int(w*radio//4)
    img[0:y, :] = 0
    img[:, w-y:w] = 0
    img[h-x:h, :] = 0
    img[:, 0:x] = 0
    return img


def main():
    # 嵌入消息
    em_flag = 1
    L = 2
    if em_flag:
        img = cv2.imread('lena.png')
        mes = tb.new_rand_bytes(L, 99)
        ste_img = embed(img, mes)
        cv2.imwrite("ste.png", ste_img)

    # 提取消息
    ex_flag = 1
    if ex_flag:
        filename = '../ste.png'
        ste_img = cv2.imread(filename)
        ste_img = cv2.resize(ste_img, (512, 512))
        mes = extract(ste_img, L)
        print("提取信息：", mes)

    pass


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
