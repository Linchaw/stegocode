
import cv2
import numpy as np
import toolbox as tb
F = 4


def embed(img, mes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pass


def main():
    # 嵌入消息
    em_flag = 1
    L = 2
    if em_flag:
        img = cv2.imread('lena.png')
        mes = tb.new_rand_bytes(L, 3)
        ste_img = embed(img, mes)
        cv2.imwrite("ste.png", ste_img)

    # 提取消息
    ex_flag = 0
    if ex_flag:
        filename = 'ste.png'
        ste_img = cv2.imread(filename)
        ste_img = cv2.resize(ste_img, (512, 512))
        # mes = extract(ste_img, L)
        # print("提取信息：", mes)

    pass


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    main()
