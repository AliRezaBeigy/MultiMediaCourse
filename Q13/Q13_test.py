#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/kodim05.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def mse(image_1, image_2):
    return ((image_1 - image_2) ** 2).sum() / (image_1.shape[0] * image_1.shape[1])


def psnr(image_1, image_2):
    mse_value = mse(image_1, image_2)
    return 10 * np.log((255 ** 2) / mse_value)


def compare(image_1, image_2):
    mse_value = mse(image_1, image_2)
    psnr_value = psnr(image_1, image_2)
    print("MSE: {:.2f} PSNR: {:.2f}".format(mse_value, psnr_value))


image_height, image_width, channels = image.shape

block_size = 8

for alpha in range(1, 10):
    image_blocks = np.empty(
        (int(image_height / block_size), int(image_width / block_size)),
        dtype=object,
    )
    block_height, block_width = image_blocks.shape

    for i in range(block_width):
        for j in range(block_height):
            image_blocks[j, i] = image[
                j * block_size : (j + 1) * block_size,
                i * block_size : (i + 1) * block_size,
                :,
            ]

    for i in range(block_width):
        for j in range(block_height):
            block = image_blocks[j, i]
            image_blocks[j, i] = [cv2.dct(block[:, :, x]) for x in range(channels)]

    # mask = np.zeros((block_size, block_size))
    # mask[0 : int(block_size / 4), :] = 1
    # mask[:, 0 : int(block_size / 4)] = 1

    # for i in range(block_width):
    #     for j in range(block_height):
    #         for c in range(channels):
    #             image_blocks[j, i][c] = mask * image_blocks[j, i][c]

    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])

    for i in range(block_width):
        for j in range(block_height):
            for c in range(channels):
                image_blocks[j, i][c] = image_blocks[j, i][c] / (quantization_matrix * alpha)

    for i in range(block_width):
        for j in range(block_height):
            for c in range(channels):
                image_blocks[j, i][c] = image_blocks[j, i][c] * (quantization_matrix * alpha)

    for i in range(block_width):
        for j in range(block_height):
            image_blocks[j, i] = cv2.merge(
                tuple([cv2.idct(image_blocks[j, i][x]) for x in range(channels)])
            )
            
    result = np.copy(image)

    for i in range(block_width):
        for j in range(block_height):
            result[
                j * block_size : (j + 1) * block_size,
                i * block_size : (i + 1) * block_size,
                :,
            ] = image_blocks[j, i]

    # plt.figure()
    # plt.imshow(result)
    # plt.figure()
    # plt.imshow(image)
    # plt.show()
    ############################################
    print('Alpha: {}'.format(alpha))
    compare(image, result)
    print()
