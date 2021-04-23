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

paper = cv2.imread(path.abspath(
    path.join(__file__, "..", "..", "images", "paper.png")))
paper = cv2.cvtColor(paper, cv2.COLOR_BGR2RGB)

# plt.figure()
# plt.imshow(paper)

paper_laplacian2 = cv2.Laplacian(paper, cv2.CV_64F).astype('uint8')

# plt.figure()
# plt.imshow(paper_laplacian)


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


def scale(image, rate):
    return image[::rate[0], ::rate[1], :]


# paper = scale(paper, (2, 2))
# plt.imshow(paper)

L = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
paper_laplacian = cv2.filter2D(paper, cv2.CV_32F, L).astype('uint8')
# paper_laplacian = correlation(paper, F)

# paper_median_blur_3 = cv2.medianBlur(paper, 3)
# paper_median_blur_5 = cv2.medianBlur(paper, 5)
# paper_median_blur_11 = cv2.medianBlur(paper, 11)

# plt.figure()
# plt.imshow(paper + (paper - paper_median_blur_3))
# plt.figure()
# plt.imshow(paper + (paper - paper_median_blur_5))
# plt.figure()
# plt.imshow(paper + (paper - paper_median_blur_11))
plt.figure()
plt.imshow(paper_laplacian)
plt.figure()
plt.imshow(paper + (paper - paper_laplacian))
# plt.figure()
# plt.imshow(paper + paper_laplacian2)

plt.show()
