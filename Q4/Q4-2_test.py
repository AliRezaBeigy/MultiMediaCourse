#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import numpy as np
from os import path
import matplotlib.pyplot as plt

motors = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "kodim05.png")))
motors = cv2.cvtColor(motors, cv2.COLOR_BGR2GRAY)

plt.figure()
plt.imshow(motors, cmap='gray')

sobel_filter_hx = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]])
sobel_filter_hy = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])


def correlation(image, kernel):
    image_width = image.shape[1]
    image_height = image.shape[0]
    kernel_height, kernel_width = kernel.shape

    result = np.zeros(image.shape)

    for c in range(1 if len(image.shape) < 3 else image.shape[2]):
        padding_image = np.zeros((image_height + kernel_height - 1,
                                  image_width + kernel_width - 1))

        if len(image.shape) < 3:
            padding_image[round((kernel_height - 1) / 2):image_height + round((kernel_height - 1) / 2),
                          round((kernel_width - 1) / 2):image_width + round((kernel_width - 1) / 2)] \
                = image[:, :]
        else:
            padding_image[round((kernel_height - 1) / 2):image_height + round((kernel_height - 1) / 2),
                          round((kernel_width - 1) / 2):image_width + round((kernel_width - 1) / 2)] \
                = image[:, :, c]

        for y in range(image_height + round((kernel_height - 1) / 2)):
            for x in range(image_width + round((kernel_width - 1) / 2)):
                sample = padding_image[y:y + kernel_height, x:x + kernel_width]
                if sample.shape != kernel.shape:
                    continue
                if len(image.shape) < 3:
                    result[y, x] = np.sum(sample * kernel)
                else:
                    result[y, x, c] = np.sum(sample * kernel)

    result = result / result.max() * 255

    return result


# correlation(np.array([[1, 1, 1, 1],
#                       [1, 1, 1, 1],
#                       [1, 1, 1, 1]]),
#             np.array([[1, 1, 1],
#                       [1, 1, 1],
#                       [1, 1, 1]]))
motors_sobel_hx = cv2.filter2D(motors, -1, sobel_filter_hx)
motors_sobel_hy = cv2.filter2D(motors, -1, sobel_filter_hy)

plt.figure()
plt.imshow(motors_sobel_hx, cmap='gray')
plt.figure()
plt.imshow(motors_sobel_hy, cmap='gray')

plt.show()
