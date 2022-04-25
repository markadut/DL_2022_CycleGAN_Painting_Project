from configparser import Interpolation
from tkinter import X
from turtle import xcor
from typing import final
# from cv2 import dft
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
from tensorflow.keras.preprocessing.image import smart_resize

def get_data(file_path, melanoma_class):
 

   ########### PREPARE DATASET TO HAVE IMAGE PATHS, IMAGE RGB VALUES (np.array) ##########

    df = pd.read_csv("data/data_ham10000/HAM10000_metadata.csv")
    # print(df.head(10))

    #handling missing data: 
    df.isnull().sum()
    df['age'].fillna(int(df['age'].mean()),inplace=True)

    #lesion types dict to map 
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    base_skin_dir = file_path

    # Merge images from both folders into one dictionary
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                        for x in glob.glob(os.path.join(base_skin_dir, '', '*.jpg'))}

    df['path'] = df['image_id'].map(imageid_path_dict.get)
    df['cell_type'] = df['dx'].map(lesion_type_dict.get) 
    df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

    #Resizing Images:
    df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((256,256))))

    ###################################################################################

    inputs = df["image"]
    inputs = inputs/(255.0) # normalization
    labels = df["cell_type"].to_list()

    interim_inputs = [j for i, j in enumerate(inputs) if (labels[i] == melanoma_class)]

    inp_reshape = tf.reshape(interim_inputs, (-1, 256, 256, 3))
    final_inputs = np.asarray(inp_reshape, dtype= np.float32)

    melanoma_images = []
    for tensor in final_inputs:
        melanoma_images.append(tensor)

    melanoma_tensor_converted = tf.convert_to_tensor(melanoma_images)
    melanoma_dataset = tf.data.Dataset.from_tensor_slices(melanoma_tensor_converted)

    ds_size = tf.data.experimental.cardinality(melanoma_dataset)
    train_split=0.8
    test_split=0.2
    shuffle_size=1113

    Shuffle=True
    if Shuffle:
    # Specify seed to always have the same split distribution between runs
        ds = melanoma_dataset.shuffle(shuffle_size, seed=12)

    train_size = int(np.ceil(train_split * int(ds_size)))
    test_size = int(np.ceil(test_split * int(ds_size)))

    train_ds = ds.take(train_size)    
    test_ds = ds.take(test_size)

    return train_ds, test_ds

















# def get_data(file_path, melanoma_class, benign_class):
 

#    ########### PREPARE DATASET TO HAVE IMAGE PATHS, IMAGE RGB VALUES (np.array) ##########

#     df = pd.read_csv("data/data_ham10000/HAM10000_metadata.csv")
#     # print(df.head(10))

#     #handling missing data: 
#     df.isnull().sum()
#     df['age'].fillna(int(df['age'].mean()),inplace=True)

#     #lesion types dict to map 
#     lesion_type_dict = {
#         'nv': 'Melanocytic nevi',
#         'mel': 'Melanoma',
#         'bkl': 'Benign keratosis-like lesions ',
#         'bcc': 'Basal cell carcinoma',
#         'akiec': 'Actinic keratoses',
#         'vasc': 'Vascular lesions',
#         'df': 'Dermatofibroma'
#     }

#     base_skin_dir = file_path

#     # Merge images from both folders into one dictionary
#     imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
#                         for x in glob.glob(os.path.join(base_skin_dir, '', '*.jpg'))}

#     df['path'] = df['image_id'].map(imageid_path_dict.get)
#     df['cell_type'] = df['dx'].map(lesion_type_dict.get) 
#     df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes

#     #Resizing Images:
#     df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))

#     ###################################################################################

#     inputs = df["image"]
#     inputs = inputs/(255.0) # normalization
#     labels = df["cell_type_idx"].to_list()

#     interim_inputs = [j for i, j in enumerate(inputs) if (labels[i] == melanoma_class or labels[i]==benign_class)]
#     interim_labels = [j for i, j in enumerate(labels) if (labels[i] == melanoma_class or labels[i]==benign_class)]

#     #binarizing labels
#     new_labels = [0 if (i == melanoma_class) else 1 for i in interim_labels]
    
#     #one-hot encoding the final labels 
#     final_labels = tf.one_hot(new_labels, depth = 2)

#     inp_reshape = tf.reshape(interim_inputs, (-1, 32, 32, 3))
#     final_inputs = np.asarray(inp_reshape, dtype= np.float32)

#     #shuffle: is it necessary? alternatives? 
#     tf.random.shuffle(final_inputs)
#     tf.random.shuffle(final_labels)

#     train_inputs = final_inputs[:1000]
#     test_inputs = final_inputs[1001:]
#     train_labels = final_labels[:1000]
#     test_labels = final_labels[1001:]
    
#     return train_inputs, test_inputs, train_labels, test_labels

# # get_data("data/data_ham10000/HAM10000_images", 1, 2)
 











