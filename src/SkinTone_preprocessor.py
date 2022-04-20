#Importing required libraries
from typing import final
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
from tensorflow.keras.preprocessing.image import array_to_img


def skin_tone_preprocessor(file_path):

    #create dataframe to store image paths and images:

    images_dict = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob.glob(os.path.join(file_path, '', '*.jpeg'))}

    df = pd.DataFrame(images_dict.items()) 
    #rename columns for clarity
    df = df.rename(columns={df.columns[0]: "image_id"})
    df = df.rename(columns={df.columns[1]: "image_path"})
    df["image"] = df['image_path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))

    imgs = df["image"]
    inputs = np.array(imgs)/(255.0) # normalization

    interim_inputs = [j for i, j in enumerate(inputs)]
    inp_reshape = tf.reshape(interim_inputs, (-1, 3, 32 ,32))

    final_inputs = np.asarray(tf.transpose(inp_reshape, perm=[0,2,3,1]), dtype= np.float32)
    
    return final_inputs


skin_tone_preprocessor("data/SkinTone_Dataset")