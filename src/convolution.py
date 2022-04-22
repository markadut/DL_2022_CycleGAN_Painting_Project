from __future__ import absolute_import
import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = inputs.shape[0]
	in_height = inputs.shape[1]
	in_width = inputs.shape[2]
	input_in_channels = inputs.shape[3]

	filter_height = filters.shape[0]
	filter_width = filters.shape[1]
	filter_in_channels = filters.shape[2]
	filter_out_channels = filters.shape[3]

	#stride of 1
	num_examples_stride = strides[0]
	strideY = strides[1]
	strideX = strides[2]
	channels_stride = strides[3]

	# ensure that the input's number of "in channels" is equivalent to the filters' number of "in channels"
	assert input_in_channels == filter_in_channels, "Filter in channels & Input in channels aren't the same length."

	
	if padding == 'SAME':
		padY = math.floor((filter_height-1)/2)
		padX = math.floor((filter_width-1)/2)

	elif padding == 'VALID':
		padY = 0
		padX = 0

	else: 
		print("Padding neither VALID nor SAME")

	# Calculate output dimensions
	out_height = math.floor((in_height + (2*padY) - filter_height) / strideY + 1)
	out_width = math.floor((in_width + (2*padX) - filter_width) / strideX + 1)

	# Generate NumPy array with the correct output dimensions, whose elements will be updated
	# as the convolution operator is performed
	final_output = np.zeros((math.floor(num_examples), out_height, out_width, math.floor(filter_out_channels)))

	#applying padding
	inputs = np.pad(inputs, ((0,0), (padY, padX), (padX, padY), (0,0)))


	#iterating through the # of examples
	for i in range(0, num_examples, num_examples_stride):
		#iterating through filter_out_channels
		for fil in range(0, filter_out_channels, channels_stride):
			current_filter = filters[:, :, :, fil]
			#iterating through output_height
			for o_h in range(0, out_height, strideY):
				#iterating through output_weight
				for o_w in range(0, out_width, strideX):
					inp = inputs[i, o_h:o_h + filter_height, o_w:o_w + filter_width, :]
					final_output[i][o_h][o_w][fil] = tf.reduce_sum(tf.multiply(inp, current_filter))


	#converting output to tensor:
	return tf.convert_to_tensor(final_output)


def same_test_0():
	'''
	Simple test using SAME padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
	print("SAME_TEST_0:", "my conv2d:", my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())

def valid_test_0():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[2,2,3,3,3],[0,1,3,0,3],[2,3,0,1,3],[3,3,2,1,2],[3,3,0,2,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,5,5,1))
	filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_0:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_1():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[3,5,3,3],[5,1,4,5],[2,5,0,1],[3,3,2,1]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
								dtype=tf.float32,
								stddev=1e-1),
								name="filters")
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def valid_test_2():
	'''
	Simple test using VALID padding to check out differences between 
	own convolution function and TensorFlow's convolution function.

	NOTE: DO NOT EDIT
	'''
	imgs = np.array([[1,3,2,1],[1,3,3,1],[2,1,1,3],[3,2,3,3]], dtype=np.float32)
	imgs = np.reshape(imgs, (1,4,4,1))
	filters = np.array([[1,2,3],[0,1,0],[2,1,2]]).reshape((3,3,1,1)).astype(np.float32)
	my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
	tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
	print("VALID_TEST_1:", "my conv2d:", my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())

def main():
	# TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output
	same_test_0()
	valid_test_0()
	# valid_test_1()
	# valid_test_2()


if __name__ == '__main__':
	main()
