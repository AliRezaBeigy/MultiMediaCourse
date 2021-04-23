#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import sys
import math
import numpy as np
from os import path
import matplotlib.pyplot as plt

image = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "messi5.jpg")))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure()
# plt.imshow(image)

ball_image = image[290:335, 338:387, :]
# plt.figure()
# plt.imshow(ball_image)

ball_image_hist = cv2.calcHist([ball_image], [0], None, [256], [0, 256])

# plt.figure()
# plt.plot(ball_image_hist)


def compareHistogram(sample, target):
    return np.sum(np.abs(sample - target))


width_padding = math.floor((image.shape[1] % ball_image.shape[1]) / 2)
height_padding = math.floor((image.shape[0] % ball_image.shape[0]) / 2)
padding_image = cv2.copyMakeBorder(
    image, height_padding, height_padding, width_padding, width_padding, cv2.BORDER_CONSTANT, value=0)

# plt.figure()
# plt.imshow(padding_image)

sample_step = 1
target_width = ball_image.shape[1]
target_height = ball_image.shape[0]

target_in_image_position = (0, 0, sys.maxsize)  # (height, width, thrashold)

for j in range(0, padding_image.shape[0], sample_step):
    for i in range(0, padding_image.shape[1], sample_step):
        sample = padding_image[j:j + target_height, i:i + target_width, :]
        sample_hist = cv2.calcHist([sample], [0], None, [256], [0, 256])

        thrashold = compareHistogram(sample_hist, ball_image_hist)

        if(thrashold < target_in_image_position[2]):
            target_in_image_position = (j, i, thrashold)

target_image = image[target_in_image_position[0]:target_in_image_position[0] +
                     target_height, target_in_image_position[1]:target_in_image_position[1] + target_width, :]

plt.figure()
plt.imshow(target_image)

plt.show()
