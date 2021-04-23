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
from skimage.util import random_noise

cameraman = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "cameraman.tif")))
cameraman = cv2.cvtColor(cameraman, cv2.COLOR_BGR2RGB)

babon = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "mandril.tiff")))
babon = cv2.cvtColor(babon, cv2.COLOR_BGR2RGB)

# f, subplt = plt.subplots(1, 2, figsize=(10, 5))
# subplt[0].imshow(cameraman)
# subplt[0].set_title("Cameraman")
# subplt[1].imshow(babon)
# subplt[1].set_title("Babon")


def log_transform(image):
    image = image * 255
    c = 255 / np.log(1 + 255)
    return c * np.log(image + 1)


def exp_transform(image):
    c = 255 / np.log(1 + 255)
    return (np.exp(image / c))


sp_noised_cameraman = log_transform(random_noise(cameraman, mode='s&p'))
gaussian_noised_cameraman = log_transform(
    random_noise(cameraman, mode='gaussian'))

sp_noised_babon = log_transform(random_noise(babon, mode='s&p'))
gaussian_noised_babon = log_transform(random_noise(babon, mode='gaussian'))

sp_noised_cameraman_mean_filer_3 = exp_transform(
    cv2.blur(sp_noised_cameraman, (3, 3)))
sp_noised_cameraman_mean_filer_5 = cv2.blur(sp_noised_cameraman, (5, 5))

sp_noised_babon_mean_filer_3 = cv2.blur(sp_noised_babon, (3, 3))
sp_noised_babon_mean_filer_5 = cv2.blur(sp_noised_babon, (5, 5))

plt.imshow(sp_noised_cameraman_mean_filer_3.astype('uint8'))

plt.show()
