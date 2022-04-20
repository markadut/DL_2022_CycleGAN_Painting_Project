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

base_skin_dir = "data/data_ham10000/HAM10000_images"

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                    for x in glob.glob(os.path.join(base_skin_dir, '', '*.jpg'))}

df['path'] = df['image_id'].map(imageid_path_dict.get)
df['cell_type'] = df['dx'].map(lesion_type_dict.get) 
df['cell_type_idx'] = pd.Categorical(df['cell_type']).codes
# print(df.head(10))

#Resizing Images:
df['image'] = df['path'].map(lambda x: np.asarray(Image.open(x).resize((100, 75))))

# visualize images:
# n_samples = 5
# fig, m_axs = plt.subplots(7, n_samples, figsize = (4*n_samples, 3*7))
# for n_axs, (type_name, type_rows) in zip(m_axs, 
#                                         df.sort_values(['cell_type']).groupby('cell_type')):
#     n_axs[0].set_title(type_name)
#     for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state=2018).iterrows()):
#         c_ax.imshow(c_row['image'])
#         c_ax.axis('off')
# save and display sample image: 
# fig.savefig('category_samples.png', dpi=300)

features=df.drop(columns=['cell_type_idx'],axis=1)
target=df['cell_type_idx']

#sample convert to tensor
img = df["image"][0]
img = tf.convert_to_tensor(img)

# convert tensor to PIL image:
pil_img = tf.keras.preprocessing.image.array_to_img(img)
pil_img.show()

data_augmentation = tf.keras.Sequential([
    RandomFlip(mode="horizontal_and_vertical"),
    RandomRotation(0.2),
])

print(df)