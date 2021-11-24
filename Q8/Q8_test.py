import cv2
import math
import numpy as np
from random import randint
import matplotlib.pyplot as plt

lena_image = cv2.imread("./Images/lena512.bmp", cv2.IMREAD_GRAYSCALE)

# sobel_filter_x = np.array([[-1, 0, 1],
#                            [-2, 0, 2],
#                            [-1, 0, 1]])
# sobel_filter_y = np.array([[-1, -2, -1],
#                            [0, 0, 0],
#                            [1, 2, 1]])


# def fft2convolve(image, kernel):
#     image_width, image_height = image.shape
#     kernel_height, kernel_width = kernel.shape
#     padding_kernel = np.zeros((image_height, image_width))

#     padding_kernel[round(image_height / 2 - (kernel_height - 1) / 2):round(image_height / 2 + (kernel_height - 1) / 2) + 1,
#                    round(image_width / 2 - (kernel_width - 1) / 2):round(image_width / 2 + (kernel_width - 1) / 2) + 1] \
#         = kernel[:, :]

#     image_freq = np.fft.fft2(image)
#     padding_kernel_freq = np.fft.fft2(padding_kernel)

#     result = np.fft.ifft2(image_freq * padding_kernel_freq)

#     return np.fft.fftshift(result).real


# frequency domain
# lena_sobel_x_image = fft2convolve(lena_image, sobel_filter_x)
# lena_sobel_y_image = fft2convolve(lena_image, sobel_filter_y)
# lena_sobel_y_x_image = fft2convolve(lena_sobel_y_image, sobel_filter_x)

# fig, subplot = plt.subplots(1, 4, figsize=(5, 5))
# plt.tight_layout()

# subplot[0].axis('off')
# subplot[0].set_title("Lena")
# subplot[0].imshow(lena_image, cmap='gray')
# subplot[1].axis('off')
# subplot[1].set_title("Sobel X")
# subplot[1].imshow(lena_sobel_x_image, cmap='gray')
# subplot[2].axis('off')
# subplot[2].set_title("Sobel Y")
# subplot[2].imshow(lena_sobel_y_image, cmap='gray')
# subplot[3].axis('off')
# subplot[3].set_title("Sobel X & Y")
# subplot[3].imshow(lena_sobel_y_x_image, cmap='gray')

# plt.show()

# spatial domain
# lena_sobel_x_image = cv2.filter2D(lena_image, cv2.CV_32F, sobel_filter_x)
# lena_sobel_y_image = cv2.filter2D(lena_image, cv2.CV_32F, sobel_filter_y)
# lena_sobel_y_x_image = cv2.filter2D(lena_sobel_y_image, cv2.CV_32F,
#                                     sobel_filter_x)

# fig, subplot = plt.subplots(1, 4, figsize=(5, 5))
# plt.tight_layout()

# subplot[0].axis('off')
# subplot[0].set_title("Lena")
# subplot[0].imshow(lena_image, cmap='gray')
# subplot[1].axis('off')
# subplot[1].set_title("Sobel X")
# subplot[1].imshow(lena_sobel_x_image, cmap='gray')
# subplot[2].axis('off')
# subplot[2].set_title("Sobel Y")
# subplot[2].imshow(lena_sobel_y_image, cmap='gray')
# subplot[3].axis('off')
# subplot[3].set_title("Sobel X & Y")
# subplot[3].imshow(lena_sobel_y_x_image, cmap='gray')

# plt.show()

# image_width, image_height = lena_image.shape
# kernel_height, kernel_width = sobel_filter_x.shape

# padding_sobel_filter_x = np.zeros((image_height, image_width))
# padding_sobel_filter_x[round(image_height / 2 - (kernel_height - 1) / 2):round(image_height / 2 + (kernel_height - 1) / 2) + 1,
#                        round(image_width / 2 - (kernel_width - 1) / 2):round(image_width / 2 + (kernel_width - 1) / 2) + 1] \
#     = sobel_filter_x[:, :]

# padding_sobel_filter_y = np.zeros((image_height, image_width))
# padding_sobel_filter_y[round(image_height / 2 - (kernel_height - 1) / 2):round(image_height / 2 + (kernel_height - 1) / 2) + 1,
#                        round(image_width / 2 - (kernel_width - 1) / 2):round(image_width / 2 + (kernel_width - 1) / 2) + 1] \
#     = sobel_filter_y[:, :]

# padding_sobel_filter_x_freq = np.fft.fft2(padding_sobel_filter_x)
# padding_sobel_filter_y_freq = np.fft.fft2(padding_sobel_filter_y)

# fig, subplot = plt.subplots(2, 4, figsize=(5, 5))
# plt.tight_layout()

# subplot[0][0].axis('off')
# subplot[0][0].set_title("Sobel X Real")
# subplot[0][0].imshow(padding_sobel_filter_x_freq.real, cmap="gray")
# subplot[0][1].axis('off')
# subplot[0][1].set_title("Sobel X Imaginary")
# subplot[0][1].imshow(padding_sobel_filter_x_freq.imag, cmap="gray")
# subplot[0][2].axis('off')
# subplot[0][2].set_title("Sobel X Absolute")
# subplot[0][2].imshow(np.absolute(padding_sobel_filter_x_freq), cmap="gray")
# subplot[0][3].axis('off')
# subplot[0][3].set_title("Sobel X Angle")
# subplot[0][3].imshow(np.angle(padding_sobel_filter_x_freq), cmap="gray")

# subplot[1][0].axis('off')
# subplot[1][0].set_title("Sobel Y Real")
# subplot[1][0].imshow(padding_sobel_filter_y_freq.real, cmap="gray")
# subplot[1][1].axis('off')
# subplot[1][1].set_title("Sobel Y Imaginary")
# subplot[1][1].imshow(padding_sobel_filter_y_freq.imag, cmap="gray")
# subplot[1][2].axis('off')
# subplot[1][2].set_title("Sobel Y Absolute")
# subplot[1][2].imshow(np.absolute(padding_sobel_filter_y_freq), cmap="gray")
# subplot[1][3].axis('off')
# subplot[1][3].set_title("Sobel Y Angle")
# subplot[1][3].imshow(np.angle(padding_sobel_filter_y_freq), cmap="gray")

# plt.show()


# def createCircleMask(shape, distance):
#     mask = np.zeros(shape)
#     rows, columns = shape
#     for x in range(columns):
#         for y in range(rows):
#             if np.sqrt((x - columns/2) ** 2 + (y - rows/2) ** 2) < distance:
#                 mask[y, x] = 1
#     return mask


# def createGaussianMask(shape, distance):
#     mask = np.zeros(shape)
#     rows, columns = shape
#     for x in range(columns):
#         for y in range(rows):
#             mask[y, x] = np.exp(
#                 ((-1 * (np.sqrt((x - columns/2) ** 2 + (y - rows/2) ** 2)))/(2*(distance**2))))
#     return mask


# def mse(image_1, image_2):
#     return ((image_1 - image_2) ** 2).sum() / \
#         (image_1.shape[0] * image_1.shape[1])


# def psnr(image_1, image_2):
#     mse_value = mse(image_1, image_2)

#     if mse_value == 0:
#         return math.inf

#     return 10 * np.log((255 ** 2) / mse_value)


# fig, subplot = plt.subplots(2, 5, figsize=(30, 10))
# plt.tight_layout()

# lena_image_freq = np.fft.fft2(lena_image)
# image_width, image_height = lena_image.shape

# for i in range(5):
#     circle_mask = createCircleMask((image_width, image_height),
#                                    image_width/(2*(5-i)))

#     lena_image_freq_shift = np.fft.fftshift(lena_image_freq)

#     lena_image_freq_mask = lena_image_freq_shift * circle_mask

#     lena_image_freq_mask = np.fft.fftshift(lena_image_freq_mask)

#     result = np.fft.ifft2(lena_image_freq_mask)

#     subplot[0][i].axis(False)
#     subplot[0][i].imshow(circle_mask, cmap="gray")
#     subplot[0][i].set_title("Mask")

#     subplot[1][i].axis(False)
#     subplot[1][i].imshow(result.real, cmap="gray")
#     subplot[1][i].set_title(f"PSNR={psnr(lena_image, result.real)}")

# plt.show()

# fig, subplot = plt.subplots(2, 5, figsize=(30, 10))
# plt.tight_layout()

# lena_image_freq = np.fft.fft2(lena_image)
# image_width, image_height = lena_image.shape

# for i in range(5):
#     x = cv2.getGaussianKernel(image_width, image_width/(2*(5-i)))
#     gaussian_mask = x * x.T

#     lena_image_freq_shift = np.fft.fftshift(lena_image_freq)

#     lena_image_freq_mask = lena_image_freq_shift * gaussian_mask

#     lena_image_freq_mask = np.fft.fftshift(lena_image_freq_mask)

#     result = np.fft.ifft2(lena_image_freq_mask)

#     subplot[0][i].axis(False)
#     subplot[0][i].imshow(gaussian_mask, cmap="gray")
#     subplot[0][i].set_title("Mask")

#     subplot[1][i].axis(False)
#     subplot[1][i].imshow(result.real, cmap="gray")
#     subplot[1][i].set_title(f"PSNR={psnr(lena_image, result.real)}")

# plt.show()


# fig, subplot = plt.subplots(2, 5, figsize=(30, 10))

# squre = np.zeros((10, 10))

# for i in range(5):
#     mask = np.zeros((100, 100))

#     x = randint(0, mask.shape[0] - squre.shape[0])
#     y = randint(0, mask.shape[1] - squre.shape[1])

#     mask[x + squre.shape[0]:x + squre.shape[0] + squre.shape[0],
#          y + squre.shape[1]:y + squre.shape[1] + squre.shape[1]] = squre

#     subplot[0][i].axis(False)
#     subplot[0][i].imshow(mask, cmap="gray")
#     subplot[0][i].set_title("Mask")

#     mask_freq = np.fft.fft2(mask)

#     subplot[1][i].axis(False)
#     subplot[1][i].imshow(mask_freq.real, cmap="gray")
#     subplot[1][i].set_title("Real")

#     subplot[2][i].axis(False)
#     subplot[2][i].imshow(mask_freq.imag, cmap="gray")
#     subplot[2][i].set_title("Imaginary")

#     subplot[3][i].axis(False)
#     subplot[3][i].imshow(np.absolute(mask_freq), cmap="gray")
#     subplot[3][i].set_title("Absolute")

#     subplot[4][i].axis(False)
#     subplot[4][i].imshow(np.angle(mask_freq), cmap="gray")
#     subplot[4][i].set_title("Angle")

# plt.show()

fig, subplot = plt.subplots(5, 4, figsize=(15, 10))

lena_image_freq = np.fft.fft2(lena_image)
image_width, image_height = lena_image.shape

for i in range(20, 100, 20):
    mask = np.zeros((100, 100))
    squre = np.ones((i, i))

    x = randint(0, mask.shape[0] - squre.shape[0])
    y = randint(0, mask.shape[1] - squre.shape[1])

    mask[round(mask.shape[0] / 2 - squre.shape[0] / 2):round(mask.shape[0] / 2 - squre.shape[0] / 2 + squre.shape[0]),
         round(mask.shape[1] / 2 - squre.shape[0] / 2):round(mask.shape[1] / 2 - squre.shape[0] / 2 + squre.shape[1])] = squre

    i = round(i / 20 - 1)
    subplot[0][i].axis(False)
    subplot[0][i].imshow(mask, cmap="gray")
    subplot[0][i].set_title("Mask")

    mask_freq = np.fft.fft2(mask)

    subplot[1][i].axis(False)
    subplot[1][i].imshow(mask_freq.real, cmap="gray")
    subplot[1][i].set_title("Real")

    subplot[2][i].axis(False)
    subplot[2][i].imshow(mask_freq.imag, cmap="gray")
    subplot[2][i].set_title("Imaginary")

    subplot[3][i].axis(False)
    subplot[3][i].imshow(np.absolute(mask_freq), cmap="gray")
    subplot[3][i].set_title("Absolute")

    subplot[4][i].axis(False)
    subplot[4][i].imshow(np.angle(mask_freq), cmap="gray")
    subplot[4][i].set_title("Angle")

plt.show()
