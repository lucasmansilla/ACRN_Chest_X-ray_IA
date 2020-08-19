import tensorflow as tf


def upsample2d(tensor, factor):
    return tf.image.resize_images(tensor,
                                  factor * tf.shape(tensor)[1:-1],
                                  tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def conv2d(x, name, dim, k, s, p, bn, af, is_train):
    with tf.variable_scope(name):
        w = tf.get_variable(
                'weights', [k, k, x.get_shape()[-1], dim],
                initializer=tf.truncated_normal_initializer(stddev=0.01))
        x = tf.nn.conv2d(x, w, [1, s, s, 1], p)

        if bn:
            x = batch_norm(x, 'bn', is_train=is_train)
        else:
            b = tf.get_variable(
                    'biases', [dim],
                    initializer=tf.constant_initializer(0.))
            x += b

        if af:
            x = af(x)

    return x


def upconv2d(x, factor, name, dim, k, s, p, bn, af, is_train):
    return conv2d(upsample2d(x, factor), name, dim, k, s, p, bn, af, is_train)


def dense(x_in, name, dim, bn, af, is_train):
    x = tf.layers.dense(inputs=x_in,
                        units=dim,
                        name=name,
                        trainable=is_train,
                        use_bias=not bn,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        activation=af)

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
