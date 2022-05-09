from matplotlib import image
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
import glob 
import os 
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf
# from skimage.measure import compare_ssim
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def structural_sim_index_generator(imageA_dir, imageB_dir):

    # 3. Load the two input images
    imageA = cv2.imread(imageA_dir)
    imageB = cv2.imread(imageB_dir)

    imageA = cv2.resize(imageA, (840, 840))
    imageB = cv2.resize(imageB, (840, 840))

    # print('Resized Dimensions A: ',imageA.shape)
    # print('Resized Dimensions B: ',imageB.shape)

    # # 4. Convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    (score, diff) = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")

    # 6. You can print only the score if you want
    print("SSIM of Lambda: {}".format(score))


lambda_0_5_ssim = structural_sim_index_generator("struct_similarity_images/sample_skin_tone.png", "struct_similarity_images/lamda_0_5_epoch_30.png")
lambda_1_ssim = structural_sim_index_generator("struct_similarity_images/sample_skin_tone.png", "struct_similarity_images/lambda_1_epoch_30.png")
lambda_10_ssim = structural_sim_index_generator("struct_similarity_images/sample_skin_tone.png", "struct_similarity_images/lambda_10_epoch_30.png")
lamda_20_ssim = structural_sim_index_generator("struct_similarity_images/sample_skin_tone.png", "struct_similarity_images/lambda_20_epoch_30.png")





# img = cv2.imread("struct_similarity_images/lambda_20_epoch_30.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# noise = np.ones_like(img) * 0.2 * (img.max() - img.min())
# rng = np.random.default_rng()
# noise[rng.random(size=noise.shape) > 0.5] *= -1

# img_noise = img + noise
# img_const = img + abs(noise)

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 4),
#                          sharex=True, sharey=True)
# ax = axes.ravel()

# mse_none = mean_squared_error(img, img)
# ssim_none = ssim(img, img, data_range=img.max() - img.min())

# mse_noise = mean_squared_error(img, img_noise)
# ssim_noise = ssim(img, img_noise,
#                   data_range=img_noise.max() - img_noise.min())

# mse_const = mean_squared_error(img, img_const)
# ssim_const = ssim(img, img_const,
#                   data_range=img_const.max() - img_const.min())

# ax[0].imshow(img, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[0].set_xlabel(f'MSE: {mse_none:.2f}, SSIM: {ssim_none:.2f}')
# ax[0].set_title('Original image')

# ax[1].imshow(img_noise, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[1].set_xlabel(f'MSE: {mse_noise:.2f}, SSIM: {ssim_noise:.2f}')
# ax[1].set_title('Image with noise')

# ax[2].imshow(img_const, cmap=plt.cm.gray, vmin=0, vmax=1)
# ax[2].set_xlabel(f'MSE: {mse_const:.2f}, SSIM: {ssim_const:.2f}')
# ax[2].set_title('Image plus constant')

# plt.tight_layout()
# plt.show()


