#Importing required libraries
from asyncio.base_tasks import _task_print_stack
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnn_model import train

def skin_tone_preprocessor(file_path):

    #create dataframe to store image paths and images:

    images_dict = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob.glob(os.path.join(file_path, '', '*.jpeg'))}

    df = pd.DataFrame(images_dict.items()) 
    #rename columns for clarity
    df = df.rename(columns={df.columns[0]: "image_id"})
    df = df.rename(columns={df.columns[1]: "image_path"})
    df["image"] = df['image_path'].map(lambda x: np.asarray(Image.open(x).resize((256, 256))))

    imgs = df["image"]
    inputs = np.array(imgs)/(255.0) # normalization

    interim_inputs = [j for i, j in enumerate(inputs)]
    inp_reshape = tf.reshape(interim_inputs, (-1, 256, 256, 3))
    final_inputs = np.asarray(inp_reshape, dtype= np.float32)
    
    train_imgs = tf.random.shuffle(final_inputs)
    
    og_images = []
    for tensor in train_imgs:
        og_images.append(tensor)

    # Apply data augmentation to populate some data 
    # With data augmentation to prevent overfitting 
    lst_saturated = []
    for i in range(len(train_imgs)):
        saturation_played_1_3 = tf.image.adjust_saturation(train_imgs[i], 1.3)
        saturation_played_1_6 = tf.image.adjust_saturation(train_imgs[i], 1.6)
        saturation_played_1_9 = tf.image.adjust_saturation(train_imgs[i], 1.9)
        lst_saturated.append(saturation_played_1_3)
        lst_saturated.append(saturation_played_1_6)
        lst_saturated.append(saturation_played_1_9)

    res_list = [y for x in [og_images, lst_saturated] for y in x]
    tensor_converted_saturation = tf.convert_to_tensor(res_list)
    saturation_skinTone_dataset = tf.data.Dataset.from_tensor_slices(tensor_converted_saturation)

    ds_size = tf.data.experimental.cardinality(saturation_skinTone_dataset)
    train_split=0.8
    test_split=0.2
    shuffle_size=296

    Shuffle=True
    if Shuffle:
    # Specify seed to always have the same split distribution between runs
        ds = saturation_skinTone_dataset.shuffle(shuffle_size, seed=12)

    train_size = int(np.ceil(train_split * int(ds_size)))
    test_size = int(np.ceil(test_split * int(ds_size)))

    train_ds = ds.take(train_size)    
    test_ds = ds.take(test_size)

    # return train_ds, test_ds
    train_size_lst = []
    for img in train_ds:
        train_size_lst.append(img)

    train_imgs_arrays = []
    for tensor_images in train_size_lst:
        array_img = np.asarray(tensor_images)
        train_imgs_arrays.append(array_img)
    
    #uncomment for visualization purposes
    # for i in range(len(train_imgs_arrays)):
    #     test_input_i = train_imgs_arrays[i]
    #     plt.figure(figsize=(12, 12))
    #     display_list = [test_input_i]
    #     title = ['Input Image']
    #     for j in range(1):
    #         plt.subplot(1, 2, j+1)
    #         plt.title(title[j])
    #         plt.imshow(display_list[j])
    #         plt.axis('off')
    #     plt.savefig(f"training_skin_tone_image_{i}")

    test_size_lst = []
    for img in test_ds:
        test_size_lst.append(img)

    test_imgs_arrays = []
    for tensor_images in test_size_lst:
        array_img = np.asarray(tensor_images)
        test_imgs_arrays.append(array_img)
    
    #uncomment for visualization purposes
    # for i in range(len(test_imgs_arrays)):
    #     test_input_i = test_imgs_arrays[i]
    #     plt.figure(figsize=(12, 12))
    #     display_list = [test_input_i]
    #     title = ['Test Input Image']
    #     for j in range(1):
    #         plt.title(title[j])
    #         plt.imshow(display_list[j])
    #         plt.axis('off')
    #     plt.savefig(f"testing_skin_tone_image_{i}")

    return train_ds, test_ds

