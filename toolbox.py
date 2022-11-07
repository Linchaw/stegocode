"""
author: hlc
function:
    pseudo_shuffle(array, seed=0)  伪随机置乱图像
    pseudo_sort(array, seed=0)  恢复伪随机置乱图像
    bytes_shuffle(data, seed=0)  伪随机置乱字节串
    bytes_sort(data, seed=0)  恢复伪随机置乱字节串
    new_rand_bytes(length=32, seed=0)  生成一个随机字节串
    bytes2binstr(data)  字节串转换为二进制字符串
    binstr2bytes(binstr)  二进制字符串转成字节串
    error_rate(str1, str2)  计算两个字符串的误码率
    jpeg_quantization_table()  生成JPEG量化表
    imshow()  显示图像
"""

import numpy as np
import cv2


def pseudo_shuffle(array, seed=0):
    """
    Pseudo shuffle array or image.
    :param array: array to be shuffled
    :param seed: seed for random number generator
    :return: shuffled array
    """
    # pseudo array initialization
    np.random.seed(seed)
    random_array = np.arange(array.shape[0] * array.shape[1])
    random_array = np.random.permutation(random_array)

    # initialize array to store shuffled values
    shuffle_array = np.zeros_like(array)

    # shuffle
    if array.ndim == 1:
        for i in range(len(array)):
            shuffle_array[i] = array[random_array[i] - 1]
    elif array.ndim >= 2:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                shuffle_array[i, j] = array[random_array[i * array.shape[1] + j] // array.shape[1], random_array[i * array.shape[1] + j] % array.shape[1]]
    else:
        print('Invalid array dimension')

    return shuffle_array


def pseudo_sort(array, seed=0):
    """
    Pseudo sort array or image.
    :param array: pseudo shuffled array
    :param seed: seed for random number generator
    :return: sorted array
    """
    # pseudo array initialization
    np.random.seed(seed)
    random_array = np.arange(array.shape[0] * array.shape[1])
    random_array = np.random.permutation(random_array)

    # initialize array to store shuffled values
    sort_array = np.zeros_like(array)

    # shuffle
    if array.ndim == 1:
        for i in range(len(array)):
            sort_array[random_array[i]-1] = array[i]
    elif array.ndim >= 2:
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                sort_array[random_array[i * array.shape[1] + j] // array.shape[1], random_array[i * array.shape[1] + j] % array.shape[1]] = array[i, j]
    else:
        print('Invalid array dimension')

    return sort_array


def bytes_shuffle(data, seed=0):
    """
    Pseudo shuffle bytes.
    :param data: origin bytes to be shuffled
    :param seed: seed for random number generator
    :return: shuffled bytes
    """
    data_length = len(data)
    np.random.seed(seed)
    random_array = np.arange(data_length)
    random_array = np.random.permutation(random_array)

    shuffle_data = bytearray(data_length)
    for i in range(data_length):
        shuffle_data[i] = data[random_array[i]]
    return bytes(shuffle_data)


def bytes_sort(data, seed=0):
    """
    Pseudo sort bytes.
    :param data: pseudo shuffled bytes
    :param seed: seed for random number generator
    :return: origin bytes
    """
    data_length = len(data)
    np.random.seed(seed)
    random_array = np.arange(data_length)
    random_array = np.random.permutation(random_array)
    sort_data = bytearray(data_length)
    for i in range(data_length):
        sort_data[random_array[i]] = data[i]
    return bytes(sort_data)


def new_rand_bytes(length=32, seed=0):
    """
    Generate new bytes.
    :param length: length of bytes
    :param seed: seed for random number generator
    :return: new bytes
    """
    np.random.seed(seed)
    random_array = np.random.randint(0, 256, length)
    data = bytearray(length)
    for i in range(length):
        data[i] = random_array[i]
    return bytes(data)


def bytes2binstr(data):
    binstr = ''
    for byte in data:
        substr = ''
        for i in range(8):
            bit = byte % 2
            substr = str(bit) + substr
            byte = byte // 2
        binstr += substr
    return binstr


def binstr2bytes(binstr):
    if len(binstr) % 8 != 0:
        print("str is not binstr format!")
        return None
    
    binlist = [binstr[i:i+8] for i in range(0, len(binstr), 8)]
    bytes_list = []
    for bins in binlist:
        byte = 0
        for bit in bins:
            byte *= 2
            byte += int(bit)
        bytes_list.append(byte)
    data = bytes(bytes_list)
    return data


def jpeg_quantization_table(quality=-1):
    """
    Generate JPEG quantization table.
    :param quality: quality of JPEG image
    :return: quantization table
    """
    table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])

    if quality < 0:
        return table

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


def error_rate(str1, str2):
    """
    Calculate error rate of two strings.
    :param str1: string 1
    :param str2: string 2
    :return: error rate
    """
    if len(str1) != len(str2):
        print('Error: two strings have different length!')
        return None
    error = 0
    for i in range(len(str1)):
        if str1[i] != str2[i]:
            error += 1
    return error / len(str1)

def imshow(image, name='image'):
    """
    Show image.
    :param image: image to be shown
    :param name: name of window
    :return: None
    """
    cv2.imshow(name, image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    """Main function."""
    # # test to shuffle bytes
    # with open('lena.png', 'rb') as f:
    #     data = f.read(32)
    # shuffle_data = bytes_shuffle(data, seed=1)
    # recover_data = bytes_sort(shuffle_data, seed=1)
    # print(data)
    # print(shuffle_data)
    # print(recover_data)

    # # test to shuffle array
    # rand_bytes = new_rand_bytes()

    # data = new_rand_bytes(8, 2)
    # binstr = bytes2binstr(data)
    # binstr = '10101000000011111110110101001000000101100010101111010010010010111'  # error length
    # Bytes = binstr2bytes(binstr)
    # print(data)
    # print(binstr)
    # print(Bytes)
    table = jpeg_quantization_table()
    print(table)
    pass


if __name__ == '__main__':
    main()
