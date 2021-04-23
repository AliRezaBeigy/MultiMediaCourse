#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
from os import path
import matplotlib.pyplot as plt
from skimage.util import random_noise

cameraman = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "cameraman.tif")))
cameraman = cv2.cvtColor(cameraman, cv2.COLOR_BGR2RGB)

# plt.figure()
# plt.imshow(cameraman)


sp_noised_cameraman = random_noise(cameraman, mode='s&p')
gaussian_noised_cameraman = random_noise(cameraman, mode='gaussian')

# plt.figure()
# plt.imshow(gaussian_noised_image)

# plt.figure()
# plt.imshow(sp_noised_image)

sp_noised_cameraman_gaussian_filer_3 = cv2.GaussianBlur(
    sp_noised_cameraman, (3, 3), cv2.BORDER_DEFAULT)
sp_noised_cameraman_gaussian_filer_5 = cv2.GaussianBlur(
    sp_noised_cameraman, (5, 5), cv2.BORDER_DEFAULT)
sp_noised_cameraman_median_filer_3 = cv2.medianBlur(
    sp_noised_cameraman.astype('float32'), 3)
sp_noised_cameraman_median_filer_5 = cv2.medianBlur(
    sp_noised_cameraman.astype('float32'), 5)
sp_noised_cameraman_mean_filer_3 = cv2.blur(sp_noised_cameraman, (3, 3))
sp_noised_cameraman_mean_filer_5 = cv2.blur(sp_noised_cameraman, (5, 5))

gaussian_noised_cameraman_gaussian_filer_3 = cv2.GaussianBlur(
    gaussian_noised_cameraman, (3, 3), cv2.BORDER_DEFAULT)
gaussian_noised_cameraman_gaussian_filer_5 = cv2.GaussianBlur(
    gaussian_noised_cameraman, (5, 5), cv2.BORDER_DEFAULT)
gaussian_noised_cameraman_median_filer_3 = cv2.medianBlur(
    gaussian_noised_cameraman.astype('float32'), 3)
gaussian_noised_cameraman_median_filer_5 = cv2.medianBlur(
    gaussian_noised_cameraman.astype('float32'), 5)
gaussian_noised_cameraman_mean_filer_3 = cv2.blur(
    gaussian_noised_cameraman, (3, 3))
gaussian_noised_cameraman_mean_filer_5 = cv2.blur(
    gaussian_noised_cameraman, (5, 5))

f, sub_plt = plt.subplots(6, 2, figsize=(16, 5))
sub_plt[0][0].imshow(sp_noised_cameraman_gaussian_filer_3)
sub_plt[0][1].set_title("Cameraman Salt and Pepper With Gaussian 3x3")
sub_plt[1][0].imshow(sp_noised_cameraman_gaussian_filer_5)
sub_plt[1][1].set_title("Cameraman Salt and Pepper With Gaussian 5x5")
sub_plt[2][0].imshow(sp_noised_cameraman_median_filer_3)
sub_plt[2][1].set_title("Cameraman Salt and Pepper With Median 3x3")
sub_plt[3][0].imshow(sp_noised_cameraman_median_filer_5)
sub_plt[3][1].set_title("Cameraman Salt and Pepper With Median 5x5")
sub_plt[4][0].imshow(sp_noised_cameraman_mean_filer_3)
sub_plt[4][1].set_title("Cameraman Salt and Pepper With Mean 3x3")
sub_plt[5][0].imshow(sp_noised_cameraman_mean_filer_5)
sub_plt[5][1].set_title("Cameraman Salt and Pepper With Mean 5x5")

plt.show()
