import tensorflow as tf
import numpy as np

from .displacement import batch_displacement_warp2d
from .ops import conv2d, upconv2d
from .utils import save_image


class VectorCNN(object):

    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x):
        non_linearity = tf.nn.elu

        with tf.variable_scope(self.name, reuse=self.reuse):
            conv1_a = conv2d(x, 'conv1_a', 16, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv1_b = conv2d(conv1_a, 'conv1_b', 16, 3, 1, 'SAME',
                             True, None, self.is_train)
            act1 = non_linearity(conv1_b)
            pool1 = tf.nn.avg_pool(act1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv2_a = conv2d(pool1, 'conv2_a', 32, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv2_b = conv2d(conv2_a, 'conv2_b', 32, 3, 1, 'SAME',
                             True, None, self.is_train)
            act2 = non_linearity(conv2_b)
            pool2 = tf.nn.avg_pool(act2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv3_a = conv2d(pool2, 'conv3_a', 64, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv3_b = conv2d(conv3_a, 'conv3_b', 64, 3, 1, 'SAME',
                             True, None, self.is_train)
            act3 = non_linearity(conv3_b)
            pool3 = tf.nn.avg_pool(act3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

            conv4_a = conv2d(pool3, 'conv4_a', 128, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv4_b = conv2d(conv4_a, 'conv4_b', 128, 3, 1, 'SAME',
                             True, None, self.is_train)
            act4 = non_linearity(conv4_b)
            drop4 = tf.layers.dropout(act4, rate=0.5,
                                      training=self.is_train,
                                      name='drop4')

            deconv4 = upconv2d(drop4, 2, 'deconv4', 64, 3, 1, 'SAME',
                               True, None, self.is_train)
            concat5 = non_linearity(tf.add(deconv4, conv3_b))
            conv5_a = conv2d(concat5, 'conv5_a', 64, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv5_b = conv2d(conv5_a, 'conv5_b', 64, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            drop5 = tf.layers.dropout(conv5_b, rate=0.5,
                                      training=self.is_train,
                                      name='drop5')

            deconv5 = upconv2d(drop5, 2, 'deconv5', 32, 3, 1, 'SAME',
                               True, None, self.is_train)
            concat6 = non_linearity(tf.add(deconv5, conv2_b))
            conv6_a = conv2d(concat6, 'conv6_a', 32, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv6_b = conv2d(conv6_a, 'conv6_b', 32, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            drop6 = tf.layers.dropout(conv6_b, rate=0.5,
                                      training=self.is_train,
                                      name='drop6')

            deconv6 = upconv2d(drop6, 2, 'deconv6', 16, 3, 1, 'SAME',
                               True, None, self.is_train)
            concat7 = non_linearity(tf.add(deconv6, conv1_b))
            conv7_a = conv2d(concat7, 'conv7_a', 16, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv7_b = conv2d(conv7_a, 'conv7_b', 16, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv_out = conv2d(conv7_b, 'conv_out', 2, 3, 1, 'SAME',
                              True, None, self.is_train)

        if self.reuse is None:
            self.var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.train.Saver(
                var_list=self.var_list, max_to_keep=100)
            self.reuse = True

        return conv_out

    def save(self, sess, ckpt_path, step):
        self.saver.save(sess, ckpt_path, global_step=step)

    def restore(self, sess, ckpt_path):
        self.saver.restore(sess, ckpt_path)


class ACRegNet(object):

    def __init__(self, sess, name, im_size):
        self.sess = sess
        self.name = name

        im_shape = [1] + im_size + [1]

        self.x = tf.placeholder(tf.float32, im_shape)
        self.y = tf.placeholder(tf.float32, im_shape)
        self.xy = tf.concat([self.x, self.y], axis=3)

        if im_shape[1:-1] != [64, 64]:
            dsfac = im_shape[1] / 64.
            x_reshaped = tf.image.resize_images(self.x, size=[64, 64])
            y_reshaped = tf.image.resize_images(self.y, size=[64, 64])

            self.xy = tf.concat([x_reshaped, y_reshaped], axis=3)

        self.VectorCNN = VectorCNN('VectorCNN', is_train=False)
        self.v = self.VectorCNN(self.xy)

        if list(im_shape[1:-1]) != [64, 64]:
            self.v = tf.image.resize_images(self.v, size=im_shape[1:-1])
            self.v = self.v * dsfac

        self.z = batch_displacement_warp2d(
            self.x, self.v, vector_fields_in_pixel_space=True)

        self.sess.run(tf.global_variables_initializer())

    def deploy(self, dir_path, x, y, save_df_info=False):
        warp_im, df = self.sess.run([self.z, self.v], {
                self.x: x,
                self.y: y})

        if dir_path is not None:
            save_image(dir_path + '/result_image.png', warp_im[0, :, :, 0])
            if save_df_info:
                np.save(dir_path + '/df.npy', np.squeeze(df))

        return warp_im, df

    def save(self, ckpt_path, step=None):
        self.VectorCNN.save(self.sess, ckpt_path + '/model.ckpt', step)

    def restore(self, ckpt_path, step=None):
        ckpt_file = 'model.ckpt'
        if step is not None:
            ckpt_file = ckpt_file + '-' + str(step)
        self.VectorCNN.restore(self.sess, ckpt_path + '/' + ckpt_file)
