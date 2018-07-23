import tensorflow as tf
import numpy as np

def conv2d(input, output_dim,
        k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
        name="conv2d", with_w = False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(
            input,
            w,
            strides = [1, d_h, d_w, 1],
            padding = 'SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        if with_w:
            return conv, w, biases
        else:
            return conv

def activate(input_tensor, activation, name=None):
        if activation == "sigmoid":
            return tf.nn.sigmoid(input_tensor, name=name)
        if activation == "tanh":
            return tf.nn.tanh(input_tensor, name=name)
        if activation == "relu":
            return tf.nn.relu(input_tensor, name=name)
        if activation == "elu":
            return tf.nn.elu(input_tensor, name=name)
        if activation == 'softmax':
            return tf.nn.softmax(input_tensor, name=name)
        else:
            assert False,'Invalid activation function {0}'.format(activation)

def max_pool(input_tensor, k_h = 2, k_w = 2, d_h = 2, d_w = 2, padding = 'SAME', name = 'MAX_POOL'):
    kernel = [1, k_h, k_w, 1]
    stride = [1, d_h, d_w, 1]
    return tf.nn.max_pool(input_tensor, kernel, stride, padding, 'NHWC', name)


""" Batch Normalization """
# post about Batch Normalization in TensorFlow
# http://ruishu.io/2016/12/27/batchnorm/
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name) as scope:
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name
            self.scope = scope

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, scope=self.name, is_training = train)

""" Linear Layer """
def linear(input, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))

        if with_w:
            return tf.matmul(input, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input, matrix) + bias

""" Loss Func"""
def loss_func(loss_func, actual, values, epsilon=1e-10):
    if loss_func == "rmse":
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(actual, values))))
    elif loss_func == "cross_entropy":
        return tf.reduce_mean(-tf.reduce_sum(
            actual * tf.log(values + epsilon) + (1 - actual) * tf.log(1 - values + epsilon), reduction_indices=[1]
        ))

    else:
        assert False, 'Invalid loss function : {}'.format(loss_func)