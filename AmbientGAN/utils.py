from tensorflow.contrib import slim
import numpy as np
from PIL import Image
import tensorflow as tf
import os, math, cv2


# arr should be 4 dimensional
def save_image(arr, name, idx, scale=True, path='./mnist/generated/'):
    if scale:
        arr = arr * (255.0 / np.max(arr))
    for i in range(arr.shape[0]):
        img_to_save = arr[i, :, :, :].astype(np.uint8)
        cv2.imwrite(path + str(idx) + '_' + str(i) + '_' + name, img_to_save)
    print('SAVING GENERATED IMAGES TO: ' + path + name)

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def batch_norm(input_op, name, is_training, epsilon=1e-5, momentum=0.99):
    return tf.contrib.layers.batch_norm(input_op, decay=momentum, updates_collections=None,
                                        epsilon=epsilon, scale=True, is_training=is_training, scope=name)

def show_all_variables():
    all_variables = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(all_variables, print_info=True)

def lrelu(input_op, leak=0.2, name='linear'):
    return tf.maximum(input_op, leak*input_op, name=name)

def conv2d(input_op, n_out, name, kh=5, kw=5, dh=2, dw=2, use_bias=True):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernel_w',
                                 shape=(kh, kw, n_in, n_out),
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        if use_bias:
            biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(0.0))
            return tf.nn.bias_add(conv, biases)
        else:
            return conv

def deconv2d(input_op, output_shape, kh=5, kw=5, dh=2, dw=2, name='deconv', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value
    n_out = output_shape[-1]
    # filter : [height, width, output_channels, in_channels]
    with tf.variable_scope(name):
        kernel = tf.get_variable(name='kernels',
                                 shape=(kh, kw, n_out, n_in),
                                 initializer=tf.contrib.layers.xavier_initializer_conv2d())
        deconv = tf.nn.conv2d_transpose(input_op, kernel,
                                        output_shape=output_shape,
                                        strides=(1, dh, dw, 1))
        biases = tf.get_variable(name='biases', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.nn.bias_add(deconv, biases)

def pooling(input_op, name, kh=2, kw=2, dh=2, dw=2, pooling_type='max'):
    if 'max' in pooling_type:
        return tf.nn.max_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)
    else:
        return tf.nn.avg_pool(input_op, ksize=[1, kh, kw, 1], strides=[1, dh, dw, 1], padding='SAME', name=name)

def fully_connect(input_op, n_out, name='fully_connected', bias_init=0.0):
    n_in = input_op.get_shape()[-1].value

    with tf.variable_scope(name):
        kernel = tf.get_variable(name='weights',
                                 shape=[n_in, n_out],
                                 dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())

        biases = tf.get_variable(name='bias', shape=(n_out), initializer=tf.constant_initializer(bias_init))
        return tf.matmul(input_op, kernel) + biases

def random_mask(input_op, batch_size=64):
    masks = [tf.ones([1, 32, 32, 3], dtype=tf.float32) for _ in range(batch_size)]
    off_h = tf.random_uniform([batch_size], 0, 32, dtype=tf.int32)
    off_w = tf.random_uniform([batch_size], 0, 32, dtype=tf.int32)
    paddings = [tf.image.pad_to_bounding_box(masks[it], off_h[it], off_w[it], 96, 96)
                for it in range(batch_size)]
    masking = 1.0 - tf.cast(tf.concat(paddings, axis=0), dtype=tf.float32)
    return input_op * masking
