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

image = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "Airplane.tiff")))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image)


def calcHist(image):
    histogram = np.zeros(256)

    for c in range(image.shape[2]):
        for h in range(image.shape[0]):
            for w in range(image.shape[1]):
                histogram[image[h, w, c]] += 1

    return histogram


# image_hist = cv2.calcHist([image], [0], None, [256], [0, 256])
image_hist = calcHist(image)

plt.figure()
plt.plot(image_hist)

image_equalized = np.array(image)
for i in range(image_equalized.shape[2]):
    image_equalized[:, :, i] = cv2.equalizeHist(image_equalized[:, :, i])

plt.figure()
plt.imshow(image_equalized)

image_equalized_hist = cv2.calcHist(
    [image_equalized], [0], None, [256], [0, 256])

plt.figure()
plt.plot(image_equalized_hist)
plt.show()
