import cv2
import numpy as np
import toolbox as tb
F = 5


def jpeg_q_table(quality=75):

    table = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]], dtype='uint8')

    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100
    if quality < 50:
        quality = 5000 / quality
    else:
        quality = 200 - quality * 2

    qtable = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        for j in range(8):
            qtable[i, j] = max(1, min(255, (table[i, j] * quality + 50) / 100))

    return qtable


def cv_show(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def block_dct(channel):
    # 转换为浮点型
    channel_32 = np.float32(channel)
    dct_channel = np.zeros(channel_32.shape, dtype=np.float32)

    # 分块进行DCT变换
    for i in range(0, channel.shape[0], 8):
        for j in range(0, channel.shape[1], 8):
            dct_channel[i:i+8, j:j+8] = cv2.dct(channel_32[i:i+8, j:j+8])

    return dct_channel


def block_idct(channel):
    # 转换为浮点型
    channel_32 = np.float32(channel)
    idct_channel = np.zeros(channel_32.shape, dtype=np.float32)

    # 分块进行DCT变换
    for i in range(0, channel.shape[0], 8):
        for j in range(0, channel.shape[1], 8):
            idct_channel[i:i+8, j:j+8] = cv2.idct(channel_32[i:i+8, j:j+8])

    return np.uint8(idct_channel)


def block_qdct(channel):
    # 转换为浮点型
    channel_32 = np.float32(channel)
    dct_channel = np.zeros(channel_32.shape, dtype=np.float32)
    qdct_channel = np.zeros(channel_32.shape, dtype=np.float32)
    # 创建量化表
    qtable = jpeg_q_table(75)

    # 分块进行DCT变换
    for i in range(0, channel.shape[0], 8):
        for j in range(0, channel.shape[1], 8):
            dct_channel[i:i+8, j:j+8] = cv2.dct(channel_32[i:i+8, j:j+8])
            qdct_channel[i:i+8, j:j+8] = np.round(dct_channel[i:i+8, j:j+8] / qtable, 0)

    qdct = np.int16(qdct_channel)

    return qdct


def norm_block(gray_q_dct, threshold=1500):
    ef = np.zeros((gray_q_dct.shape[0] // 8, gray_q_dct.shape[1] // 8), dtype=np.float32)
    rf = np.zeros(ef.shape, dtype=np.float32)

    for i in range(0, gray_q_dct.shape[0], 8):
        for j in range(0, gray_q_dct.shape[1], 8):
            # 计算块的能量
            ef[i // 8, j // 8] = np.sum(np.abs(gray_q_dct[i:i + 8, j:j + 8])) - np.abs(gray_q_dct[i, j])
            # 计算块的方差
            for n in range(8):
                for m in range(8):
                    if gray_q_dct[i + n, j + m] != 0:
                        rf[i // 8, j // 8] += 1

    Si = ef + rf + ef*rf

    norm = np.zeros(Si.shape)
    for i in range(0, Si.shape[0]):
        for j in range(0, Si.shape[1]):
            if Si[i, j] > threshold:
                norm[i, j] = 1

    return norm


def feature_block(gray):
    feature = np.zeros((gray.shape[0]//8, gray.shape[1]//8), dtype=np.float32)
    # sift
    # sift = cv2.xfeatures2d.SIFT_create()
    # kp = sift.detect(gray)

    # orb
    orb = cv2.ORB_create()
    kp = orb.detect(gray, None)

    for sub_kp in kp:
        x, y = sub_kp.pt
        feature[int(x/8), int(y/8)] = 1

    return feature


def modify_b(dct_b, dct_g, sm, m='1', f=F):
    dct_b_m = dct_b.copy()
    for i in range(0, dct_b.shape[0], 8):
        for j in range(0, dct_b.shape[1], 8):
            if sm[i // 8, j // 8] == 1:
                cB = 0
                cG = 0
                for si in range(f):
                    cB += dct_b[i + f - 1 - si, j + si]
                    cG += dct_g[i + f - 1 - si, j + si]

                dB = cB - cG
                if dB <= 0:
                    cB = cB + 1.3 * abs(dB) + f
                    for si in range(f):
                        dct_b_m[i + f - 1 - si, j + si] = cB / f

    return dct_b_m


def modify_m(dct_b, dct_g, sm, m=b'', f=F):
    mes = tb.bytes2binstr(m)
    print("嵌入信息：", mes)
    idx = 0
    dct_b_m = dct_b.copy()
    for i in range(0, dct_b.shape[0], 8):
        for j in range(0, dct_b.shape[1], 8):
            if sm[i // 8, j // 8] == 1:
                cB = 0
                cG = 0
                for si in range(f):
                    cB += dct_b[i + f - 1 - si, j + si]
                    cG += dct_g[i + f - 1 - si, j + si]
                dB = cB - cG

                sub_mes = mes[idx]
                if sub_mes == '1' and dB <= 0:
                    cB = cB + 1.3 * abs(dB) + f
                    for si in range(f):
                        dct_b_m[i + f - 1 - si, j + si] = cB / f

                elif sub_mes == '0' and dB > 0:
                    cB = cB - 1.3 * abs(dB) - f
                    for si in range(f):
                        dct_b_m[i + f - 1 - si, j + si] = cB / f
                idx += 1
                if idx >= len(mes):
                    return dct_b_m

    return dct_b_m


def getinfo(qdct_b, qdct_g, sm, length, f=F):
    info = ''
    for i in range(0, qdct_b.shape[0], 8):
        for j in range(0, qdct_b.shape[1], 8):
            if sm[i // 8, j // 8] == 1:
                cB = 0
                cG = 0
                for si in range(f):
                    cB += qdct_b[i + f-1 - si, j + si]
                    cG += qdct_g[i + f-1 - si, j + si]
                if cB > cG:
                    info += '1'
                else:
                    info += '0'
                    # print(i//8, j//8)
    return info[:length*8]


def embed(img, mes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    # 获取嵌入块
    gray_q_dct = block_qdct(gray)
    norm = norm_block(gray_q_dct)
    feature = feature_block(gray)
    Sm = norm * feature
    # print(np.sum(norm), np.sum(feature), np.sum(Sm))
    # 483.0 683.0 114.0

    # 嵌入信息
    dct_b = block_dct(b)
    dct_g = block_dct(g)
    # dct_b_m = modify_b(dct_b, dct_g, Sm, m)
    dct_b_m = modify_m(dct_b, dct_g, Sm, mes)

    # 分块逆dct变换
    b = block_idct(dct_b_m)
    ste_img = cv2.merge([b, g, r])

    return ste_img, (norm, feature, Sm)
    pass


def extract(ste_img, length):
    ste_gray = cv2.cvtColor(ste_img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(ste_img)

    # 获取嵌入块
    gray_q_dct = block_qdct(ste_gray)
    norm = norm_block(gray_q_dct)
    feature = feature_block(ste_gray)
    Sm = norm * feature

    # 提取信息
    dct_b = block_dct(b)
    dct_g = block_dct(g)
    info = getinfo(dct_b, dct_g, Sm, length)

    return info, (norm, feature, Sm)
    pass


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
    if 1:
        img = cv2.imread('lena.png')
        mes = tb.new_rand_bytes(length=3)
        ste_img, _ = embed(img, mes)
        cv2.imwrite("ste.png", ste_img)

    # 提取消息
    if 1:
        ste_img = cv2.imread("ste.png")
        # # 裁剪20%的边缘图像
        # ste_img = crop(ste_img, 0.2)

        ste_img = cv2.resize(ste_img, (512, 512))
        info, _ = extract(ste_img, length=3)
        print("提取信息：", info)
    pass


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
