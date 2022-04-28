from random import sample
from pandas_datareader import test
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
BATCH_SIZE = 10
IMG_WIDTH = 256
IMG_HEIGHT = 256

train_skin_tone, test_skin_tone = skin_tone_preprocessor("data/SkinTone_Dataset") 
train_melanoma, test_melanoma = get_data("data/data_ham10000/HAM10000_images", "Melanoma")

train_skin_tone = train_skin_tone.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_melanoma = train_melanoma.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_skin_tone = test_skin_tone.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_melanoma = test_melanoma.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# #####################################################################

# ########################### APPLY TF PIX2PIX ########################
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

sample_skin_patch = next(iter(train_skin_tone))
sample_melanoma = next(iter(train_melanoma))

to_melanoma = generator_g(sample_skin_patch)
to_skin_tone = generator_f(sample_melanoma)

plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_skin_patch, to_melanoma, sample_melanoma, to_skin_tone]
title = ['Skin Tone', 'To Melanoma', 'Melanoma', 'To Skin Tone']

# visualize U-net filter applied images
# for i in range(len(imgs)):
#   plt.subplot(2, 2, i+1)
#   plt.title(title[i])
#   if i % 2 == 0:
#     plt.imshow(imgs[i][0] * 0.5 + 0.5)
#   else:
#     plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
# plt.show()

# # Visualize pixel mapping
# plt.figure(figsize=(8, 8))
# plt.subplot(121)
# plt.title('Is a real melanoma lesion?')
# plt.imshow(discriminator_y(sample_melanoma)[0, ..., -1], cmap='RdBu_r')
# plt.subplot(122)
# plt.title('Is a real skin tone sample?')
# plt.imshow(discriminator_x(sample_skin_patch)[0, ..., -1], cmap='RdBu_r')
# plt.show()
# #####################################################################


# ########################### DEFINE LOSS FUNCTIONS ###################
LAMBDA = 5
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
EPOCHS = 10

def generate_images(model, test_input):
  # print("prediction", prediction.shape)
  prediction = model(test_input)
  for i in range(len(test_input)):
    prediction_i = prediction[i]
    test_input_i = test_input[i]
    plt.figure(figsize=(12, 12))
    display_list = [test_input_i, prediction_i]
    title = ['Input Image', 'Predicted Image']
    for j in range(2):
      plt.subplot(1, 2, j+1)
      plt.title(title[j])
      # getting the pixel values between [0, 1] to plot it.
      plt.imshow(display_list[j] * 0.5 + 0.5)
      plt.axis('off')
    plt.savefig(f"epoch_{epoch}_lambda_{LAMBDA}_prediction_{i}")
  plt.show()

#Training Steps: 
# * Get the predictions.
# * Calculate the loss.
# * Calculate the gradients using backpropagation.
# * Apply the gradients to the optimizer.

@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.
    
    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)
    
    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    # print("total cycle loss: ", total_cycle_loss)
    
    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)
    # print("total generator loss (adversarial + cycle): ", total_gen_g_loss)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)
    # print("discriminator loss: ", disc_x_loss)

  
  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss, 
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss, 
                                        generator_f.trainable_variables)
  
  discriminator_x_gradients = tape.gradient(disc_x_loss, 
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss, 
                                            discriminator_y.trainable_variables)
  
  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients, 
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients, 
                                            generator_f.trainable_variables))
  
  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))
  
  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))


for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_skin_tone, train_melanoma)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_skin_patch)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))

