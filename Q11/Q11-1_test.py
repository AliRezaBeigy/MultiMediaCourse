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
from skimage.util import random_noise

messi_image = cv2.imread("images/messi5.jpg")
messi_image = cv2.cvtColor(messi_image, cv2.COLOR_BGR2RGB)
barbara_image = cv2.imread("images/barbara.tif")
barbara_image = cv2.cvtColor(barbara_image, cv2.COLOR_BGR2RGB)


def mse(image_1, image_2):
    return ((image_1 - image_2) ** 2).sum() / \
        (image_1.shape[0] * image_1.shape[1])


def psnr(image_1, image_2):
    mse_value = mse(image_1, image_2)
    return 10 * np.log((255**2) / mse_value)


def compare(image_1, image_2):
    mse_value = mse(image_1, image_2)
    psnr_value = psnr(image_1, image_2)
    print("MSE: {:.2f} PSNR: {:.2f}".format(mse_value, psnr_value))


sp_noised_messi = random_noise(messi_image, mode="s&p")
gaussian_noised_messi = random_noise(messi_image, mode="gaussian")

sp_noised_barbara = random_noise(barbara_image, mode="s&p")
gaussian_noised_barbara = random_noise(barbara_image, mode="gaussian")


def applyDctTransform(block_size, image):
    image_height, image_width, channels = image.shape

    image_blocks = np.empty(
        (int(image_height / block_size), int(image_width / block_size)),
        dtype=object,
    )

    block_height, block_width = image_blocks.shape
    for i in range(block_width):
        for j in range(block_height):
            image_blocks[j,
                         i] = image[j * block_size:(j + 1) * block_size,
                                    i * block_size:(i + 1) * block_size, :, ]

    for i in range(block_width):
        for j in range(block_height):
            block = image_blocks[j, i]
            image_blocks[j, i] = [
                cv2.dct(block[:, :, x]) for x in range(channels)
            ]

    mask = np.zeros((block_size, block_size))
    mask[0:int(block_size / 2), 0:int(block_size / 2)] = 1

    for i in range(block_width):
        for j in range(block_height):
            for c in range(channels):
                image_blocks[j, i][c] = mask * image_blocks[j, i][c]

    for i in range(block_width):
        for j in range(block_height):
            image_blocks[j, i] = cv2.merge(
                tuple([
                    cv2.idct(image_blocks[j, i][x]) for x in range(channels)
                ]))

    result = np.copy(image)

    for i in range(block_width):
        for j in range(block_height):
            result[j * block_size:(j + 1) * block_size,
                   i * block_size:(i + 1) * block_size, :, ] = image_blocks[j,
                                                                            i]

    # plt.figure()
    # plt.imshow(result)
    # plt.figure()
    # plt.imshow(image)
    ############################
    print("Block Size: {}".format(block_size))
    compare(image, result)


block_sizes = [64, 16, 8]

print("Salt & Pepper Messi:")
for block_size in block_sizes:
    applyDctTransform(block_size, sp_noised_messi)
print()
print("Gaussian Messi:")
for block_size in block_sizes:
    applyDctTransform(block_size, gaussian_noised_messi)
print()
print("Salt & Pepper Barbara:")
for block_size in block_sizes:
    applyDctTransform(block_size, sp_noised_barbara)
print()
print("Gaussian Barbara:")
for block_size in block_sizes:
    applyDctTransform(block_size, gaussian_noised_barbara)

plt.show()