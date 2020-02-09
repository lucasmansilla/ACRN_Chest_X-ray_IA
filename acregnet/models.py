import tensorflow as tf
from displacement import batch_displacement_warp2d
from ops import conv2d, dense, upconv2d, l2_loss, ncc, \
    softmax_cross_entropy, total_variation
from utils import get_one_hot_encoding_from_hard_segm, \
    swap_neighbor_labels_with_prob, save_image, \
    get_hard_segm_from_prob_map
import medpy.io.save
import os
import numpy as np


class CNN_AE(object):

    def __init__(self, name, is_train):
        self.name = name
        self.is_train = is_train
        self.reuse = None

    def __call__(self, x, n_channels, n_codes):
        non_linearity = tf.nn.relu

        with tf.variable_scope(self.name, reuse=self.reuse):
            conv1_a = conv2d(x, 'conv1_a', 16, 3, 2, 'SAME',
                             True, non_linearity, self.is_train)
            conv1_b = conv2d(conv1_a, 'conv1_b', 16, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv2_a = conv2d(conv1_b, 'conv2_a', 32, 3, 2, 'SAME',
                             True, non_linearity, self.is_train)
            conv2_b = conv2d(conv2_a, "conv2_b", 32, 3, 1, 'SAME',
                             True, non_linearity, self.is_train)
            conv3 = conv2d(conv2_b, 'conv3', 1, 3, 2, 'SAME',
                           True, non_linearity, self.is_train)

            fc1 = dense(tf.reshape(conv3, [-1, np.prod([8, 8, 1])]),
                        'fc1', n_codes, False, None, self.is_train)
            fc2 = dense(fc1, 'fc2', np.prod([8, 8, 1]),
                        False, non_linearity, self.is_train)

            deconv4 = upconv2d(tf.reshape(fc2, [-1] + [8, 8, 1]), 2,
                               'deconv4', 32, 3, 1, 'SAME',
                               True, non_linearity, self.is_train)
            conv4 = conv2d(deconv4, 'conv4', 32, 3, 1, 'SAME',
                           True, non_linearity, self.is_train)
            deconv5 = upconv2d(conv4, 2, 'deconv5', 16, 3, 1, 'SAME',
                               True, non_linearity, self.is_train)
            conv5 = conv2d(deconv5, 'conv5', 16, 3, 1, 'SAME',
                           True, non_linearity, self.is_train)
            deconv6 = upconv2d(conv5, 2, 'deconv6', 16, 3, 1, 'SAME',
                               True, non_linearity, self.is_train)
            conv_out = conv2d(deconv6, 'conv_out', n_channels, 3, 1, 'SAME',
                              False, None, self.is_train)

        if self.reuse is None:
            self.var_list = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
            self.saver = tf.train.Saver(
                var_list=self.var_list, max_to_keep=100)
            self.reuse = True

        return fc1, conv_out

    def save(self, sess, ckpt_path, step):
        self.saver.save(sess, ckpt_path, global_step=step)

    def restore(self, sess, ckpt_path):
        self.saver.restore(sess, ckpt_path)


class AENet(object):

    def __init__(self, sess, config, name, is_train):
        self.sess = sess
        self.name = name
        self.is_train = is_train

        input_shape = [config['batch_size']] + \
            config['image_size'] + [config['n_labels'] + 1]

        self.x = tf.placeholder(tf.float32, input_shape)
        self.y = tf.placeholder(tf.float32, input_shape)

        self.CNN_AE = CNN_AE('CNN_AE', is_train=self.is_train)

        fc1, conv_out = self.CNN_AE(self.x, input_shape[-1], config['n_codes'])
        self.h = fc1
        self.z = conv_out

        if self.is_train:
            self.loss = softmax_cross_entropy(labels=self.y, logits=self.z)
            self.optim = tf.train.AdamOptimizer(config['learning_rate'])
            self.train = self.optim.minimize(
                self.loss, var_list=self.CNN_AE.var_list)

        self.sess.run(tf.global_variables_initializer())

    def fit(self, train_set):
        _, loss = self.sess.run(
            [self.train, self.loss],
            {
                self.x: get_one_hot_encoding_from_hard_segm(
                    swap_neighbor_labels_with_prob(train_set)),
                self.y: get_one_hot_encoding_from_hard_segm(train_set)})
        return loss

    def deploy(self, dir_path, test_set):
        z = get_hard_segm_from_prob_map(
            self.sess.run(
                self.z,
                {self.x: get_one_hot_encoding_from_hard_segm(test_set)}))
        if dir_path is not None:
            for i in range(z.shape[0]):
                save_image(
                    os.path.join(
                        dir_path, '{:02d}_x.png'.format(i + 1)),
                    test_set[i, :, :, 0], is_integer=True)
                save_image(
                    os.path.join(
                        dir_path, '{:02d}_y.png'.format(i + 1)),
                    z[i, :, :, 0], is_integer=True)
        return z

    def get_codes(self, test_set):
        return self.sess.run(
            self.h,
            {self.x: get_one_hot_encoding_from_hard_segm(test_set)})

    def save(self, ckpt_path, step=None):
        self.CNN_AE.save(
            self.sess, os.path.join(ckpt_path, 'model.ckpt'), step)

    def restore(self, ckpt_path, step=None):
        ckpt_file = 'model.ckpt'
        if step is not None:
            ckpt_file = ckpt_file + '-' + str(step)
        self.CNN_AE.restore(
            self.sess, os.path.join(ckpt_path, ckpt_file))


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

    def __init__(self, sess, config, name, is_train):
        self.sess = sess
        self.name = name
        self.is_train = is_train

        im_shape = [config['batch_size']] + config['image_size'] + [1]

        self.x = tf.placeholder(tf.float32, im_shape)
        self.y = tf.placeholder(tf.float32, im_shape)
        self.xy = tf.concat([self.x, self.y], axis=3)

        if self.is_train:
            lb_shape = [config['batch_size']] + \
                config['image_size'] + [config['n_labels'] + 1]

            self.xlabel = tf.placeholder(tf.float32, lb_shape)
            self.ylabel = tf.placeholder(tf.float32, lb_shape)
        else:
            if im_shape[1:-1] != [64, 64]:
                dsfac = im_shape[1] / 64.
                x_reshaped = tf.image.resize_images(self.x, size=[64, 64])
                y_reshaped = tf.image.resize_images(self.y, size=[64, 64])

                self.xy = tf.concat([x_reshaped, y_reshaped], axis=3)

        self.VectorCNN = VectorCNN('VectorCNN', is_train=self.is_train)
        self.v = self.VectorCNN(self.xy)

        if self.is_train:
            self.z = batch_displacement_warp2d(
                self.x, self.v, vector_fields_in_pixel_space=True)
            self.zlabel = batch_displacement_warp2d(
                self.xlabel, self.v, vector_fields_in_pixel_space=True)

            self.CNN_AE = CNN_AE('CNN_AE', is_train=False)

            with tf.name_scope('AE_1'):
                h1, _ = self.CNN_AE(
                    self.ylabel, lb_shape[-1], config['n_codes'])
            with tf.name_scope('AE_2'):
                h2, _ = self.CNN_AE(
                    self.zlabel, lb_shape[-1], config['n_codes'])

            self.loss = -ncc(self.y, self.z) + \
                config['tv_reg'] * total_variation(self.v) + \
                config['ce_reg'] * softmax_cross_entropy(
                    labels=self.ylabel, logits=self.zlabel) + \
                config['ae_reg'] * l2_loss(h1, h2)
            self.optim = tf.train.AdamOptimizer(config['learning_rate'])
            self.train = self.optim.minimize(
                self.loss, var_list=self.VectorCNN.var_list)

            self.sess.run(tf.global_variables_initializer())
            self.CNN_AE.restore(
                self.sess, os.path.join(config['aenet_dir'], 'model.ckpt'))
        else:
            if list(im_shape[1:-1]) != [64, 64]:
                self.v = tf.image.resize_images(self.v, size=im_shape[1:-1])
                self.v = self.v * dsfac

            self.z = batch_displacement_warp2d(
                self.x, self.v, vector_fields_in_pixel_space=True)

            self.sess.run(tf.global_variables_initializer())

    def fit(self, batch_images_x, batch_images_y,
            batch_labels_x, batch_labels_y):
        _, loss = self.sess.run(
            [self.train, self.loss],
            {
                self.x: batch_images_x,
                self.y: batch_images_y,
                self.xlabel: get_one_hot_encoding_from_hard_segm(
                    batch_labels_x),
                self.ylabel: get_one_hot_encoding_from_hard_segm(
                    batch_labels_y)})
        return loss

    def deploy(self, dir_path, x, y, save_def_field_info=False):
        z, df = self.sess.run(
            [self.z, self.v],
            {
                self.x: x,
                self.y: y})

        if dir_path is not None:
            for i in range(z.shape[0]):
                save_image(
                    os.path.join(
                        dir_path, '{:02d}_x.png'.format(i + 1)),
                    x[i, :, :, 0])
                save_image(
                    os.path.join(
                        dir_path, '{:02d}_y.png'.format(i + 1)),
                    y[i, :, :, 0])
                save_image(
                    os.path.join(
                        dir_path, '{:02d}_z.png'.format(i + 1)),
                    z[i, :, :, 0])
                if save_def_field_info:
                    medpy.io.save(
                        np.squeeze(np.transpose(df[i, :, :, 0])),
                        os.path.join(
                            dir_path, '{:02d}_defx_U.nii.gz'.format(i + 1)),
                        hdr=False, force=True)
                    medpy.io.save(
                        np.squeeze(np.transpose(df[i, :, :, 1])),
                        os.path.join(
                            dir_path, '{:02d}_defx_V.nii.gz'.format(i + 1)),
                        hdr=False, force=True)
                    grad_magn = np.sqrt(np.multiply(df[i, :, :, 0],
                                                    df[i, :, :, 0]) +
                                        np.multiply(df[i, :, :, 1],
                                                    df[i, :, :, 1]))
                    grad_magn = np.transpose(grad_magn)
                    medpy.io.save(
                        np.squeeze(grad_magn),
                        os.path.join(
                            dir_path, '{:02d}_defx_grad.nii.gz'.format(i + 1)),
                        hdr=False, force=True)

        return z, df

    def save(self, ckpt_path, step=None):
        self.VectorCNN.save(
            self.sess, os.path.join(ckpt_path, 'model.ckpt'), step)

    def restore(self, ckpt_path, step=None):
        ckpt_file = 'model.ckpt'
        if step is not None:
            ckpt_file = ckpt_file + '-' + str(step)
        self.VectorCNN.restore(
            self.sess, os.path.join(ckpt_path, ckpt_file))
