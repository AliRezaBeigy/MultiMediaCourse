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
from functools import reduce
import matplotlib.pyplot as plt

image_1 = cv2.imread("images/HDR/StLouisArchMultExpCDR.jpg")
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image_2 = cv2.imread("images/HDR/StLouisArchMultExpEV+1.51.jpg")
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image_3 = cv2.imread("images/HDR/StLouisArchMultExpEV+4.09.jpg")
image_3 = cv2.cvtColor(image_3, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image_4 = cv2.imread("images/HDR/StLouisArchMultExpEV-1.82.jpg")
image_4 = cv2.cvtColor(image_4, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
image_5 = cv2.imread("images/HDR/StLouisArchMultExpEV-4.72.jpg")
image_5 = cv2.cvtColor(image_5, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

images = [image_1, image_2, image_3, image_4, image_5]

image_height, image_width, channels = image_1.shape

def mse(image_1, image_2):
    return ((image_1 - image_2) ** 2).sum() / \
        (image_1.shape[0] * image_1.shape[1])

def psnr(image_1, image_2):
    mse_value = mse(image_1, image_2)
    return 10 * np.log((255 ** 2) / mse_value)

def compare(image_1, image_2):
    mse_value = mse(image_1, image_2)
    psnr_value = psnr(image_1, image_2)
    print("MSE: {:.2f} PSNR: {:.2f}".format(mse_value, psnr_value))

result = image_5
for i in range(len(images) - 1):
    result = (result + images[i]) / 2

plt.figure()
plt.imshow(result)

print("Method 1")
for i in range(len(images)):
    print('Image {}:'.format(i + 1))
    compare(np.uint8(images[i] * 255), np.uint8(result * 255))

LLs = np.empty((len(images), channels), dtype=object)
LHs = np.empty((len(images), channels), dtype=object)
HLs = np.empty((len(images), channels), dtype=object)
HHs = np.empty((len(images), channels), dtype=object)

for i in range(len(images)):
    for c in range(channels):
        LLs[i, c], (LHs[i, c], HLs[i, c], HHs[i, c]) = pywt.dwt2(
            result[:, :, c], "haar"
        )

LL = np.empty(channels, dtype=object)
LH = np.empty(channels, dtype=object)
HL = np.empty(channels, dtype=object)
HH = np.empty(channels, dtype=object)

for c in range(channels):
    LL[c], (LH[c], HL[c], HH[c]) = np.mean(LLs[:, c], axis=0), (
        reduce(lambda x, y: np.maximum(x, y), LHs[:, c]),
        reduce(lambda x, y: np.maximum(x, y), HLs[:, c]),
        reduce(lambda x, y: np.maximum(x, y), HHs[:, c]),
    )

result = np.zeros(image_1.shape)

for c in range(channels):
    result[:, :, c] = pywt.idwt2((LL[c], (LH[c], HL[c], HH[c])), "haar")

result = cv2.merge(tuple([result[:, :, c] for c in range(channels)]))

print()
print("Method 2")
for i in range(len(images)):
    print('Image {}:'.format(i + 1))
    compare(np.uint8(images[i] * 255), np.uint8(result * 255))

plt.figure()
plt.imshow(result)
plt.show()
