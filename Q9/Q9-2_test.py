#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/mandril.tiff")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def mse(image_1, image_2):
    return (
        (image_1 - image_2)**2).sum() / (image_1.shape[0] * image_1.shape[1])


def psnr(image_1, image_2):
    mse_value = mse(image_1, image_2)
    return 10 * np.log((255**2) / mse_value)


def compare(image_1, image_2):
    mse_value = mse(image_1, image_2)
    psnr_value = psnr(image_1, image_2)
    print("MSE: {:.2f} PSNR: {:.2f}".format(mse_value, psnr_value))


def sharpness(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


block_sizes = [64, 16, 8]


def process(image, block_size_index):
    image_height, image_width, channels = image.shape
    block_size = block_sizes[block_size_index]

    image_blocks = np.empty(
        (int(image_height / block_size), int(image_width / block_size)),
        dtype=object,
    )
    sharpnesses = np.copy(image_blocks)
    block_height, block_width = image_blocks.shape

    for i in range(block_width):
        for j in range(block_height):
            image_blocks[j,
                         i] = image[j * block_size:(j + 1) * block_size,
                                    i * block_size:(i + 1) * block_size, :, ]

    for i in range(block_width):
        for j in range(block_height):
            sharpnesses[j, i] = sharpness(image_blocks[j, i])

    target_sharpness = np.sort(
        sharpnesses.reshape((block_height * block_width)))[::-1][int(
            block_height * block_width * 0.1)]
    sharpnesses = sharpnesses > target_sharpness

    mask = np.zeros((block_size, block_size))
    mask[0:int(block_size / 2), 0:int(block_size / 2)] = 1

    for i in range(block_width):
        for j in range(block_height):
            block = image_blocks[j, i]

            if sharpnesses[j, i]:
                image_blocks[j, i] = process(block, block_size_index + 1)
                continue

            block = np.float32(block) / 255.0
            image_blocks[j, i] = [
                cv2.dct(block[:, :, x]) for x in range(channels)
            ]

            for c in range(channels):
                image_blocks[j, i][c] = mask * image_blocks[j, i][c]

            image_blocks[j, i] = cv2.merge(
                tuple([
                    cv2.idct(image_blocks[j, i][x]) for x in range(channels)
                ]))
            image_blocks[j, i] = np.uint8(image_blocks[j, i] * 255.0)

    result = np.copy(image)

    for i in range(block_width):
        for j in range(block_height):
            result[j * block_size:(j + 1) * block_size,
                   i * block_size:(i + 1) * block_size, :, ] = image_blocks[j,
                                                                            i]

    return result


t0 = time.time()
result = process(image, 0)
t1 = time.time()

print("time: {}".format(t1 - t0))
plt.figure()
plt.imshow(result)
plt.figure()
plt.imshow(image)
compare(image, result)
plt.show()
