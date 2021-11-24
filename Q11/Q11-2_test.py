#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import pywt
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


def applyWaveleteHaarTransform(image):
    image_height, image_width, channels = image.shape

    result = np.copy(image)

    thresholds = [0.5, 0, -0.5]
    for threshold in thresholds:
        for c in range(channels):
            LL1, (LH1, HL1, HH1) = pywt.dwt2(result[:, :, c], "haar")

            LH1 *= LH1 > threshold
            HL1 *= HL1 > threshold
            HH1 *= HH1 > threshold

            result[:, :, c] = pywt.idwt2((LL1, (LH1, HL1, HH1)), "haar")

        # plt.figure()
        # plt.imshow(result)
        # plt.figure()
        # plt.imshow(image)
        ############################################
        print('Threshold: {}'.format(threshold))
        compare(image, result)


print("Salt & Pepper Messi:")
applyWaveleteHaarTransform(sp_noised_messi)
print()
print("Gaussian Messi:")
applyWaveleteHaarTransform(gaussian_noised_messi)
print()
print("Salt & Pepper Barbara:")
applyWaveleteHaarTransform(sp_noised_barbara)
print()
print("Gaussian Barbara:")
applyWaveleteHaarTransform(gaussian_noised_barbara)

plt.show()