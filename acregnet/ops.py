import tensorflow as tf
import os
import numpy as np


def repeat(tensor, repeats):
    """ Code extracted from: https://github.com/tensorflow/tensorflow/issues/8246
    """
    with tf.variable_scope('repeat'):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
        #repeated_tensor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
        #repeated_tensor = tf.reshape(tiled_tensor, [a * b for a, b in zip(tensor.get_shape(), repeats)])
        repeated_tensor = tf.reshape(tiled_tensor, [a * b for a, b in zip(tensor.shape.as_list(), repeats)])

    return repeated_tensor


def upsample2d(tensor, factor):
    return repeat(tensor, [1, factor, factor, 1])


def conv2d(x, name, dim, k, s, p, bn, af, is_train):
  with tf.variable_scope(name):
    w = tf.get_variable('weights', [k, k, x.get_shape()[-1], dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.01))
    x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

    if bn:
        x = batch_norm(x, 'bn', is_train=is_train)
    else :
        b = tf.get_variable('biases', [dim],
                            initializer=tf.constant_initializer(0.))
        x += b

    if af:
        x = af(x)

    return x
  

def upconv2d(x, factor, name, dim, k, s, p, bn, af, is_train):
    return conv2d(upsample2d(x, factor), name, dim, k, s, p, bn, af, is_train)    


def dense(x_in, name, dim, bn, af, is_train):
    x = tf.layers.dense(inputs=x_in, units =dim, name=name, trainable=is_train, use_bias=not bn,
                        kernel_initializer=tf.glorot_uniform_initializer(), activation=af)

    if bn:
        with tf.variable_scope(name):
            x = batch_norm(x, 'bn', is_train=is_train)

    return x


def batch_norm(x, name, momentum=0.9, epsilon=1e-5, is_train=True):
    return tf.contrib.layers.batch_norm(x, 
                                        decay=momentum,
                                        updates_collections=None,
                                        epsilon=epsilon,
                                        scale=True,
                                        is_training=is_train,
                                        scope=name)


def ncc(x, y):
    """ Code extracted from: https://github.com/iwyoo/DIRNet-tensorflow
    """
    mean_x = tf.reduce_mean(x, [1,2,3], keepdims=True)
    mean_y = tf.reduce_mean(y, [1,2,3], keepdims=True)
    mean_x2 = tf.reduce_mean(tf.square(x), [1,2,3], keepdims=True)
    mean_y2 = tf.reduce_mean(tf.square(y), [1,2,3], keepdims=True)
    stddev_x = tf.reduce_sum(tf.sqrt(mean_x2 - tf.square(mean_x)), [1,2,3], keepdims=True)
    stddev_y = tf.reduce_sum(tf.sqrt(mean_y2 - tf.square(mean_y)), [1,2,3], keepdims=True)
    return tf.reduce_mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def softmax_cross_entropy(labels, logits):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits))


def total_variation(y):
    return tf.reduce_mean(tf.image.total_variation(y))


def l2_loss(x, y):
    return tf.reduce_mean(tf.reduce_sum(tf.square(x - y), axis=-1))