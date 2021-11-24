#######################################################
#                                                     #
#                This file is test                    #
#       maybe contains wrong logical operation        #
#                ignore this file                     #
#                                                     #
#######################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/barbara.tif")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)\
    .astype(np.float32) / 255.0

image_height, image_width = image.shape

secret_image = np.ones(image.shape)
secret_image[image_height // 4:image_height - (image_height // 4),
             image_width // 4:image_width - (image_width // 4)] = 0

image_dct = np.uint32(np.uint32(cv2.dct(image))) * 1000
secret_image_dct = np.uint32(np.uint8(cv2.dct(secret_image)))

result_dct = image_dct + secret_image_dct

result = cv2.idct(np.float32(np.int32(result_dct)))

plt.imshow(result, cmap='gray')
plt.show()

result_dct = np.uin32(cv2.dct(result))
secret_image_dct_d = np.fmod(result_dct, 1000)
barbara_image_dct = (result_dct - math.fmod(result_dct, 1000)) / 1000

secret_image = cv2.idct(secret_image_dct_d)
barbara_image = cv2.idct(barbara_image_dct)

f, subplt = plt.subplots(1, 2, figsize=(10, 10))
plt.tight_layout()
subplt[0].set_title('The secret image')
subplt[0].imshow(secret_image, cmap='gray')
subplt[0].axis('off')
subplt[1].set_title('Barbara image')
subplt[1].imshow(barbara_image, cmap='gray')
subplt[1].axis('off')

plt.show()
