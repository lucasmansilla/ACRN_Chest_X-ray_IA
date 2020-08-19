import tensorflow as tf

from displacement import batch_displacement_warp2d
from utils import to_one_hot, to_hard_seg


class SimpleWarper(object):
    """Implementation of an image and label map warper."""

    def __init__(self, sess, batch_size, im_size, n_labels, n_channels=1,
                 vector_fields_in_pixel_space=True):
        im_shape = [batch_size] + im_size + [n_channels]
        lb_shape = [batch_size] + im_size + [n_labels + 1]
        df_shape = [batch_size] + im_size + [2]

        self.sess = sess

        self.im = tf.placeholder(tf.float32, im_shape)
        self.lb = tf.placeholder(tf.float32, lb_shape)
        self.df = tf.placeholder(tf.float32, df_shape)

        self.im_warper = batch_displacement_warp2d(
            self.im, self.df, vector_fields_in_pixel_space)
        self.lb_warper = batch_displacement_warp2d(
            self.lb, self.df, vector_fields_in_pixel_space)

    def warp_image(self, images, flow):
        """Warp images using bilinear interpolation.

        Args:
            images: 4-D numpy array of shape [batch, height, width, channels].
            flow: 4-D numpy array of shape [batch, height, width, 2].

        Returns:
            warp: 4-D numpy array of shape [batch, height, width, channels]
                containing the warped images.
        """
        warp = self.sess.run(self.im_warper, {self.im: images, self.df: flow})

        return warp

    def warp_label(self, labels, flow):
        """Warp label maps using bilinear interpolation.

        To warp the input segmentations, first they are converted to one-hot
        format, and then they are deformed via interpolation, which is
        applied per channel.

        Args:
            labels: 4-D numpy array of shape [batch, height, width, 1].
            flow: 4-D numpy array of shape [batch, height, width, 2].

        Returns:
            warp: 4-D numpy array of shape [batch, height, width, 1]
                containing the warped label maps.
        """
        labels = to_one_hot(labels)

        warp = self.sess.run(self.lb_warper, {self.lb: labels, self.df: flow})
        warp = to_hard_seg(warp)

        return warp
