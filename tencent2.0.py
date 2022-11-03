
import cv2
import numpy as np
import toolbox as tb
F = 3


def cv_show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def feature_block(gray):
    # # sift
    # sift = cv2.xfeatures2d.SIFT_create(300)
    # kp = sift.detect(gray)

    # orb
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
    return tuple(bkp)


def modify_m(mes, b, g, res, f=F):
    mes_str = tb.bytes2binstr(mes)
    print("嵌入信息：", mes_str)
    idx = 0
    bm = b.copy()
    for kp in res:
        xf, yf = kp.pt
        x, y = int(xf), int(yf)
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        b_dct = cv2.dct(np.float32(b_block))
        g_dct = cv2.dct(np.float32(g_block))
        cB, cG = (0, 0)
        for i in range(f):
            cB += b_dct[f-1-i, i]
            cG += g_dct[f-1-i, i]
        dB = cB - cG
        sub_mes = mes_str[idx]
        flag = 0
        if sub_mes == '1' and dB <= 0:
            flag = 1
            cB = cB + 1.3 * abs(dB) + f
            for i in range(f):
                b_dct[f - 1 - i, i] = cB / f

        elif sub_mes == '0' and dB > 0:
            flag = 1
            cB = cB - 1.3 * abs(dB) - f
            for i in range(f):
                b_dct[f - 1 - i, i] = cB / f

        if flag:
            b_idct = cv2.idct(b_dct)
            bm[x - 4:x + 4, y - 4:y + 4] = np.uint8(b_idct)

        idx += 1

        if idx >= len(mes_str) - 1:
            return b

    if idx < len(mes_str) - 1:
        print("未嵌入所有数据")
    return b


def get_info(res, b, g, length, f=F):
    info = ''
    for kp in res:
        xf, yf = kp.pt
        x, y = int(xf), int(yf)
        b_block = b[x - 4:x + 4, y - 4:y + 4]
        g_block = g[x - 4:x + 4, y - 4:y + 4]
        b_dct = cv2.dct(np.float32(b_block))
        g_dct = cv2.dct(np.float32(g_block))
        cB, cG = (0, 0)
        for i in range(f):
            cB += b_dct[f - 1 - i, i]
            cG += g_dct[f - 1 - i, i]
        dB = cB - cG

        if dB > 0:
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
    res = mes_block(gray, kp, 300)

    rlist = list(res)
    rlist.sort(key=lambda x: x.response, reverse=True)
    res = tuple(rlist[0:len(mes) * 8])

    # 嵌入
    bm = modify_m(mes, b, g, res, F)
    ste_img = cv2.merge([bm, g, r])
    return ste_img, res


def extract(ste_img, length):
    ste_gray = cv2.cvtColor(ste_img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(ste_img)

    kp = feature_block(ste_gray)
    res = mes_block(ste_gray, kp, 300)

    rlist = list(res)
    rlist.sort(key=lambda x: x.response, reverse=True)
    res = tuple(rlist[0:length*8])

    info = get_info(res, b, g, length, F)
    return info, res


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
        mes = tb.new_rand_bytes(L, 2)
        ste_img, res1 = embed(img, mes)
        # cv2.imwrite("ste.png", ste_img)

    # 提取消息
    ex_flag = 1
    if ex_flag:
        # ste_img = cv2.imread("ste.png")
        # ste_img = cv2.resize(ste_img, (512, 512))
        mes, res2 = extract(ste_img, L)
        print("提取信息：", mes)

    pass


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
