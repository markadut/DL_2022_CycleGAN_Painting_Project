from __future__ import absolute_import
from matplotlib import pyplot as plt
from alternative_preprocessor import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main
        #TODO
        # play around with learning rate see how it impact 
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3) 
        

        # TODO: Initialize all hyperparameters
        ''' Weight variables should be initialized from a normal distribution, std --> 0.1''' 
        
        #Initialiing the weights of the convolutional layers
        self.cw1 = tf.Variable(tf.random.truncated_normal(shape = [5, 5, 3, 16], stddev = 0.1, dtype = tf.float32))
        self.cw2 = tf.Variable(tf.random.truncated_normal(shape = [5, 5, 16, 20], stddev = 0.1, dtype = tf.float32))
        self.cw3 = tf.Variable(tf.random.truncated_normal(shape = [3, 3, 20, 20], stddev = 0.1, dtype = tf.float32))

        #Initialiing the biases of the convolutional layers
        self.cb1 = tf.Variable(tf.random.truncated_normal(shape = [16], stddev = 0.1, dtype = tf.float32))
        self.cb2 = tf.Variable(tf.random.truncated_normal(shape = [20], stddev = 0.1, dtype = tf.float32))
        self.cb3 = tf.Variable(tf.random.truncated_normal(shape = [20], stddev = 0.1, dtype = tf.float32))

        #initializing the weights of the dense (fully-connected) layers
        self.d1 = tf.Variable(tf.random.truncated_normal(shape = [320, 120], stddev = 0.1, dtype = tf.float32))
        self.d2 = tf.Variable(tf.random.truncated_normal(shape = [120, 120], stddev = 0.1, dtype = tf.float32))
        self.d3 = tf.Variable(tf.random.truncated_normal(shape = [120 ,2], stddev = 0.1, dtype = tf.float32))

        #initializing the biases of the dense (fully_connected) layers
        self.db1 = tf.Variable(tf.random.truncated_normal(shape = [120], stddev = 0.1, dtype = tf.float32))
        self.db2 = tf.Variable(tf.random.truncated_normal(shape = [120], stddev = 0.1, dtype = tf.float32))
        self.db3 = tf.Variable(tf.random.truncated_normal(shape = [2], stddev = 0.1, dtype = tf.float32))

        # TODO: Initialize all trainable parameters


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)


        #1st convolutional layer:
        conv_layer_1 = tf.nn.conv2d(inputs, self.cw1, strides = [1, 2, 2, 1], padding = "SAME")
        conv_layer_1_wb = tf.nn.bias_add(conv_layer_1, self.cb1)
        #computation of mean and variances using tf.nn.moments
        mean_conv1, var_conv1 = tf.nn.moments(conv_layer_1_wb, axes = [0 ,1, 2])
        batch_normalized_conv1 = tf.nn.batch_normalization(conv_layer_1_wb, mean_conv1, var_conv1, None, None, variance_epsilon=1e-5)
        #RELU Layer 1 
        ReLU_conv1 = tf.nn.relu(batch_normalized_conv1)
        #Max Pooling 1 - kernels of 3x3
        max_pool1 = tf.nn.max_pool(ReLU_conv1, [1,3,3,1], [1,2,2,1], padding="SAME")


        #2nd convolutional layer:
        conv_layer_2 = tf.nn.conv2d(max_pool1, self.cw2, strides = [1, 2, 2, 1], padding = "SAME")
        conv_layer_2_wb = tf.nn.bias_add(conv_layer_2, self.cb2)
        #computation of mean and variances using tf.nn.moments
        mean_conv2, var_conv2 = tf.nn.moments(conv_layer_2_wb, axes = [0 ,1, 2])
        batch_normalized_conv2 = tf.nn.batch_normalization(conv_layer_2_wb, mean_conv2, var_conv2, None, None, variance_epsilon=1e-5)
        #RELU Layer 2
        ReLU_conv2 = tf.nn.relu(batch_normalized_conv2)
        #Max Pooling 2 - kernels of 2x2
        max_pool2 = tf.nn.max_pool(ReLU_conv2, [1,2,2,1], [1,1,1,1], padding="SAME")
        

        #3rd convolutional layer:
        if is_testing: #if testing, 3rd convolutional layer must be generated using custom conv2d 
            conv_layer_3 = conv2d(max_pool2, self.cw3, strides = [1, 1, 1, 1], padding = "SAME")
            conv_layer_3 = tf.cast(conv_layer_3, tf.float32)
        else: #if testing false, tf's nn conv2d can be used
            conv_layer_3 = tf.nn.conv2d(max_pool2, self.cw3, [1, 1, 1, 1], padding = "SAME")

        
        conv_layer_3_wb = tf.nn.bias_add(conv_layer_3, self.cb3)
        conv_3_moments = tf.nn.moments(conv_layer_3_wb, axes = [0 ,1, 2])
        batch_normalized_conv3 = tf.nn.batch_normalization(conv_layer_3_wb, conv_3_moments[0], conv_3_moments[1], None, None, variance_epsilon=1e-5)
        #RELU Layer 3 
        ReLU_conv3 = tf.nn.relu(batch_normalized_conv3)
        ReLU_conv3_reshaped = tf.reshape(ReLU_conv3, [ReLU_conv3.shape[0], -1])


        #incorporating the fully-connected layers (dense layers)
        #dense layer 1
        final_1 = tf.matmul(ReLU_conv3_reshaped, self.d1) + self.db1
        final_1 = tf.nn.relu(final_1)
        # Dropout Layer 1 with rate 0.3 
        final_1 = tf.nn.dropout(final_1, rate = 0.2)   


        final_2 = tf.matmul(final_1, self.d2) + self.db2
        final_2 = tf.nn.relu(final_2)
        # Dropout Layer 2 with rate 0.3
        final_2 = tf.nn.dropout(final_2, rate = 0.2)   


        final_logits = tf.matmul(final_2, self.d3) + self.db3
        return final_logits
        


    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))



    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    batch_size = model.batch_size

    #shuffling indices
    shuffled_indices = tf.random.shuffle(np.arange(0, train_inputs.shape[0]))

    #gathering the labels and inputs
    gathered_labels = tf.gather(train_labels, shuffled_indices) 
    gathered_inputs = tf.gather(train_inputs, shuffled_indices) 


    for i in range(0, train_inputs.shape[0], batch_size):
        inpts = gathered_inputs[i:i+batch_size]
        lbls = gathered_labels[i:i+batch_size]

        #applying tf.image.random_flip_left_right on inputs to increase accuracy
        inpts = tf.image.random_flip_left_right(inpts)

        #Call the model's forward pass and calculate the loss within the scope of tf.GradientTape
        with tf.GradientTape() as tape:
            #retrieving logits from model.call()
            loss = model.loss(model.call(inpts, is_testing=False), lbls)
            model.loss_list.append(loss)

        # Leveraging the keras Models' class has the computed property trainable_variables to conveniently
        # return all the trainable variables you'd want to adjust based on the gradients
        grads = tape.gradient(loss, model.trainable_variables)  # (from lab)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return model.loss_list 


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    model_test_acc = model.accuracy(model.call(test_inputs, is_testing = False), test_labels)
    return model_test_acc
    

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label): 
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle("{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)
        
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images): 
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0): 
            correct.append(i)
        else: 
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():

    #store classes and corresponding labels
    malignant_class = 1
    malignant_label = "Melanoma"
    benign_class = 2
    benign_label = "Benign keratosis-like lesions "

    # Read in HAM10000 data (limited to 2 classes)
    train_inputs, test_inputs, train_labels, test_labels = get_data('data/data_ham10000/HAM10000_images', malignant_class, benign_class)
    
    #initialize the model
    model = Model()

    #loss calculations
    loss = []
    for i in range(25):
        loss.append(train(model, train_inputs, train_labels))

    test_acc = test(model, test_inputs, test_labels).numpy()

    #printing losses and test accuracy
    print("Test Accuracy", test_acc)

    #visualizaiton:
    logits = model.call(train_inputs, is_testing = False)
    visualize_results(train_inputs[:50], logits, train_labels[:50], malignant_label, benign_label)


if __name__ == '__main__':
    main()
