import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from SkinTone_preprocessor import skin_tone_preprocessor
from lesion_preprocessor import get_data

import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output

############################ CYCLEGAN #################################


############################ INPUT PIPELINE ############################
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

train_skin_tone, test_skin_tone = skin_tone_preprocessor("data/SkinTone_Dataset") 
train_melanoma, test_melanoma = get_data("data/data_ham10000/HAM10000_images", "Melanoma")

sample_skin_patch = next(iter(train_skin_tone))
sample_skin_patch = tf.expand_dims(sample_skin_patch, axis = 0)
sample_melanoma = next(iter(train_melanoma))
sample_melanoma = tf.expand_dims(sample_melanoma, axis = 0)

#####################################################################

########################### APPLY TF PIX2PIX ########################
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

to_melanoma = generator_g(sample_skin_patch)
to_skin_tone = generator_f(sample_melanoma)

plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_skin_patch, to_melanoma, sample_melanoma, to_skin_tone]
title = ['Skin Tone', 'To Melanoma', 'Melanoma', 'To Skin Tone']

#visualize U-net filter applied images
for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()

#Visualize pixel mapping
plt.figure(figsize=(8, 8))
plt.subplot(121)
plt.title('Is a real melanoma lesion?')
plt.imshow(discriminator_y(sample_melanoma)[0, ..., -1], cmap='RdBu_r')
plt.subplot(122)
plt.title('Is a real skin tone sample?')
plt.imshow(discriminator_x(sample_skin_patch)[0, ..., -1], cmap='RdBu_r')
plt.show()
#####################################################################


########################### DEFINE LOSS FUNCTIONS ###################
LAMBDA = 10
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss * 0.5

def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))
  
  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

########################### TRAINING THE CYCLEGAN ###################
