import tensorflow as tf


def ncc(x, y):
    """Normalized cross correlation."""
    x_mean = tf.reduce_mean(x, [1, 2, 3], keepdims=True)
    y_mean = tf.reduce_mean(y, [1, 2, 3], keepdims=True)

    x2_mean = tf.reduce_mean(x**2, [1, 2, 3], keepdims=True)
    y2_mean = tf.reduce_mean(y**2, [1, 2, 3], keepdims=True)

    x_std = tf.sqrt(x2_mean - x_mean**2)
    y_std = tf.sqrt(y2_mean - y_mean**2)

    return tf.reduce_mean((x - x_mean) * (y - y_mean) / (x_std * y_std))


def ce(labels, logits):
    """Cross entropy (with softmax)."""
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits))


def tv(y):
    """Total variation."""
    return tf.reduce_mean(tf.image.total_variation(y))


def l2(x, y):
    """Squared L2 (Euclidean) distance."""
    return tf.reduce_mean(tf.reduce_sum((x - y) ** 2, axis=-1))
