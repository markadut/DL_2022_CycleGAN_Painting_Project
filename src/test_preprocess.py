#Importing required libraries
from cv2 import dft
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import numpy as np
import pandas as pd
import os
from tensorflow.keras.utils import to_categorical
from glob import glob
from ThreadedFileLoader.ThreadedFileLoader import *
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
import cv2 
import os
from tensorflow.keras.preprocessing.image import smart_resize

def image_loader(folder):
    input_dims = (64,64)
    img_lst = []
    for x in glob.glob(os.path.join(folder, '', '*.jpg')):
        images = np.asarray(Image.open(x).resize(input_dims))
        img_lst.append(images)
    img_lst = np.asarray(img_lst)
    img_lst = img_lst/(255.0)
    return img_lst

inputs = image_loader("data/data_ham10000/HAM10000_images")

#  plot target image
for i in range(3):
    plt.subplot(2, 3, 1 + 3 + i)
    plt.axis('off')
    plt.imshow(inputs[i])
plt.show()
