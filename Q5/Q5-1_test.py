#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import math
import random
import numpy as np
from os import path

cameraman = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "cameraman.tif")))
cameraman = cv2.cvtColor(cameraman, cv2.COLOR_BGR2RGB)


def mse(image_1, image_2):
    return ((image_1 - image_2) ** 2).sum() / \
        (image_1.shape[0] * image_1.shape[1])


def psnr(image_1, image_2):
    mse_value = mse(image_1, image_2)

    if mse_value == 0:
        return math.inf

    return 10 * np.log((255 ** 2) / mse_value)


print("MSE: " + str(mse(cameraman, cameraman)))
print("PSNR: " + str(psnr(cameraman, cameraman)))

edited_cameraman = np.copy(cameraman)
for x in range(10):
    width = round(random.triangular(0, cameraman.shape[1]))
    height = round(random.triangular(0, cameraman.shape[0]))
    edited_cameraman[height, width, :] = [255, 255, 255]


print("MSE: " + str(mse(edited_cameraman, cameraman)))
print("PSNR: " + str(psnr(edited_cameraman, cameraman)))

moved_cameraman = np.copy(cameraman)

moved_cameraman[:, 0, :] = cameraman[:, cameraman.shape[1] - 1, :]
moved_cameraman[:, 1:moved_cameraman.shape[1], :] \
    = cameraman[:, 0:cameraman.shape[1] - 1, :]


print("MSE: " + str(mse(moved_cameraman, cameraman)))
print("PSNR: " + str(psnr(moved_cameraman, cameraman)))
