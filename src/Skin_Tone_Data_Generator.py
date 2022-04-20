#Importing required libraries
import matplotlib.pyplot as plt
from PIL import Image
import os
from glob import glob
from ThreadedFileLoader.ThreadedFileLoader import *
from PIL import Image

test_image = Image.open('/Users/markadut/Downloads/osfstorage-archive/af01.tif')
# test_image.show() # opens the tiff image. this rainbow color tiff

#store image size:
width, height = test_image.size
#note 3120 x 3120 

#pixels to crop 
left = 1380
top = 850
right = 1700
bottom = 1150

#cropping for image 10: 
right_new = 1600


image_path = "/Users/markadut/Desktop/CSCI2470-DeepLearning/DL_Project_Skin/SkinTone_Dataset/"
base_skin_dir = "/Users/markadut/Downloads/osfstorage-archive"

count = 0
for img_dirc in glob.glob(os.path.join(base_skin_dir, '', '*.tif')):
    save_dir = image_path
    im = Image.open(img_dirc)
    #handling image #5 eyebrows
    if count == 9:
        im = im.crop((left, top, right_new, bottom))
    else: 
        im = im.crop((left, top, right, bottom))
    # if count not in images_not_to_include:
    im.save(f'{save_dir}/new_image_{count}.jpeg')
    count += 1


    



    



    


  



