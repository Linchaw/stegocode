import cv2
import numpy as np

F = 4


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
    q_dct_channel = dct_channel.copy()
    # channel = channel - 128

    # 分块进行DCT变换
    for i in range(0, channel.shape[0], 8):
        for j in range(0, channel.shape[1], 8):
            dct_channel[i:i+8, j:j+8] = cv2.dct(channel_32[i:i+8, j:j+8])
            q_dct_channel[i:i+8, j:j+8] = np.round(dct_channel[i:i+8, j:j+8] / table)

    return q_dct_channel


def block_iqdct(qdct_channel):
    dct_channel = np.zeros(qdct_channel.shape, dtype=np.float32)
    ichannel = np.zeros(qdct_channel.shape, dtype=np.float32)
    for i in range(0, qdct_channel.shape[0], 8):
        for j in range(0, qdct_channel.shape[1], 8):
            dct_channel[i:i+8, j:j+8] = qdct_channel[i:i+8, j:j+8] * table
            ichannel[i:i+8, j:j+8] = cv2.idct(dct_channel[i:i+8, j:j+8])

    # ichannel = ichannel + 128
    return np.uint8(ichannel)


def norm_block(gray_q_dct):
    ef = np.zeros((gray_q_dct.shape[0] // 8, gray_q_dct.shape[1] // 8), dtype=np.float32)
    rf = np.zeros(ef.shape, dtype=np.float32)

    for i in range(0, gray_q_dct.shape[0], 8):
        for j in range(0, gray_q_dct.shape[1], 8):
            ef[i // 8, j // 8] = np.sum(np.abs(gray_q_dct[i:i + 8, j:j + 8])) - np.abs(gray_q_dct[i, j])
            for n in range(8):
                for m in range(8):
                    if gray_q_dct[i + n, j + m] != 0:
                        rf[i // 8, j // 8] += 1

    Si = ef + rf + ef*rf

    norm = np.zeros(Si.shape)
    for i in range(0, Si.shape[0]):
        for j in range(0, Si.shape[1]):
            if Si[i, j] > 1500:
                norm[i, j] = 1

    return norm


def feature_block(gray):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray)
    feature = np.zeros((gray.shape[0]//8, gray.shape[1]//8), dtype=np.float32)
    for sub_kp in kp:
        # print(sub_kp.pt, sub_kp.size, sub_kp.octave, sub_kp.response)
        x, y = sub_kp.pt
        feature[int(x/8), int(y/8)] = 1

    # img = cv2.drawKeypoints(gray, kp, gray)
    # plt.imshow(img), plt.show()

    return feature


def modify_b(qdct_b, qdct_g, S, m='1', f=F):
    qdct_b_m = qdct_b.copy()
    for i in range(0, qdct_b.shape[0], 8):
        for j in range(0, qdct_b.shape[1], 8):
            # 获取 f = 6 的 cB和cG
            if S[i//8, j//8] == 1:
                cB = 0
                cG = 0
                for si in range(f):
                    cB += qdct_b[i + f - 1 - si, j + si]
                    cG += qdct_g[i + f - 1 - si, j + si]

                dB = cB - cG
                if dB <= 0:
                    cB = cB + abs(dB) + 0.3 * abs(cB) + 10
                    for si in range(f):
                        qdct_b_m[i + f - 1 - si, j + si] = cB / f

    return qdct_b_m


def getinfo(qdct_b, qdct_g, S, f=F):
    info = ''
    for i in range(0, qdct_b.shape[0], 8):
        for j in range(0, qdct_b.shape[1], 8):
            if S[i // 8, j // 8] == 1:
                cB = 0
                cG = 0
                for si in range(f):
                    cB += qdct_b[i + f-1 - si, j + si]
                    cG += qdct_g[i + f-1 - si, j + si]
                if cB > cG:
                    info += '1'
                else:
                    info += '0'
    return info


def embed():
    img = cv2.imread('lena_75.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(img)

    # 获取嵌入块
    gray_q_dct = block_qdct(gray)
    norm = norm_block(gray_q_dct)
    feature = feature_block(gray)
    S = norm * feature
    # print(np.sum(norm), np.sum(feature), np.sum(S))
    # 483.0 683.0 114.0

    # 嵌入信息
    qdct_b = block_qdct(b)
    qdct_g = block_qdct(g)
    qdct_b_m = modify_b(qdct_b, qdct_g, S)

    # x = qdct_b_m - qdct_b
    # print(np.sum(np.abs(x)))
    # 分块逆dct变换
    b = block_iqdct(qdct_b_m)
    ste_img = cv2.merge([b, g, r])
    # cv_show(ste_img)
    cv2.imwrite("ste_lena_75.jpg", ste_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
    pass


def extract():
    ste_img = cv2.imread("ste_lena_75.jpg")
    # ste_gray = cv2.cvtColor(ste_img, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(ste_img)

    # 获取嵌入块
    # graydct = block_dct(ste_gray)
    # norm = norm_block(graydct)
    # feature = feature_block(ste_gray)

    S = np.ones((8, 8))
    # print(np.sum(norm), np.sum(feature), np.sum(S))
    # 482.0 684.0 114.0

    # 提取信息
    dct_b = block_dct(b)
    dct_g = block_dct(g)
    info = getinfo(dct_b, dct_g, S)
    print(info)
    pass


def test():
    img = cv2.imread('lena_75.jpg')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = img[256-32:256+32, 256-32:256+32]
    b, g, r = cv2.split(img2)
    dct_b = block_dct(b)
    dct_g = block_dct(g)
    s = np.ones((8, 8))
    m_dct_b = modify_b(dct_b, dct_g, s)
    idct_b = block_idct(m_dct_b)
    ste_img = cv2.merge([idct_b, g, r])
    cv2.imwrite("ste_lena_75.jpg", ste_img, [cv2.IMWRITE_JPEG_QUALITY, 50])
    ste_img = cv2.imread("ste_lena_75.jpg")
    sb, sg, sr = cv2.split(ste_img)
    dct_sb = block_dct(sb)
    dct_sg = block_dct(sg)
    info = getinfo(dct_sb, dct_sg, s)
    print(info.count('0'))

    pass


def main():
    # embed()
    # extract()
    test()
    pass


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
