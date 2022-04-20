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

def get_data(file_path, melanoma_class, benign_class):
    """
	Given a file path and two target classes, returns an array of 
	normalized inputs (images) and an array of labels. 
	You will want to first extract only the data that matches the 
	corresponding classes we want (there are 10 classes and we only want 2).
	You should make sure to normalize all inputs and also turn the labels
	into one hot vectors using tf.one_hot().
	Note that because you are using tf.one_hot() for your labels, your
	labels will be a Tensor, while your inputs will be a NumPy array. This 
	is fine because TensorFlow works with NumPy arrays.
	:param file_path: file path for inputs and labels, something 
	like 'CIFAR_data_compressed/train'
	:param melanoma_class:  an integer (0-9) representing the first target
	class in the CIFAR10 dataset, for a cat, this would be a 3
	:param melanoma_class:  an integer (0-9) representing the second target
	class in the CIFAR10 dataset, for a dog, this would be a 5
	:return: normalized NumPy array of inputs and tensor of labels, where 
	inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels 
	has size (num_examples, num_classes)
	""" 

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
    df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((32, 32))))

    ###################################################################################

    inputs = df["image"]
    inputs = np.array(inputs)/(255.0) # normalization
    labels = df["cell_type_idx"].to_list()

    interim_inputs = [j for i, j in enumerate(inputs) if (labels[i] == melanoma_class or labels[i]==benign_class)]
    interim_labels = [j for i, j in enumerate(labels) if (labels[i] == melanoma_class or labels[i]==benign_class)]
    
    #binarizing labels
    new_labels = [0 if (i == melanoma_class) else 1 for i in interim_labels]
    
    #one-hot encoding the final labels 
    final_labels = tf.one_hot(new_labels, depth = 2)

    # #reshaping per described by assignment doc walkthrough
    inp_reshape = tf.reshape(interim_inputs, (-1, 3, 32 ,32))
    final_inputs = np.asarray(tf.transpose(inp_reshape, perm=[0,2,3,1]), dtype= np.float32)

    #shuffle:
    tf.random.shuffle(final_inputs)
    tf.random.shuffle(final_labels)

    train_inputs = final_inputs[:1000]
    test_inputs = final_inputs[1001:]
    train_labels = final_labels[:1000]
    test_labels = final_labels[1001:]
    
    return train_inputs, test_inputs, train_labels, test_labels

get_data("data/data_ham10000/HAM10000_images", 1, 2)
    













