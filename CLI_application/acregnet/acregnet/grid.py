# This code is based on https://github.com/Ryo-Ito/spatial_transformer_network

import tensorflow as tf


def mgrid(*args, **kwargs):
    low = kwargs.pop("low", -1)
    high = kwargs.pop("high", 1)
    low = tf.to_float(low)
    high = tf.to_float(high)
    coords = (tf.linspace(low, high, arg) for arg in args)
    grid = tf.stack(tf.meshgrid(*coords, indexing='ij'))
    return grid


def batch_mgrid(n_batch, *args, **kwargs):
    grid = mgrid(*args, **kwargs)
    grid = tf.expand_dims(grid, 0)
    grids = tf.tile(grid, [n_batch] + [1 for _ in range(len(args) + 1)])
    return grids
