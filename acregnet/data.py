import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


class DataHandler(object):

    def _load_data(im_fnames, add_channel_dim=True):
        im0 = cv2.imread(im_fnames[0], 0)
        im_batch = np.zeros((len(im_fnames),) + im0.shape)
        im_batch[0] = im0
        for i, fname in enumerate(im_fnames[1:], 1):
            im_batch[i] = cv2.imread(fname, 0)

        if add_channel_dim:
            return np.expand_dims(im_batch, axis=-1)

        return im_batch

    @staticmethod
    def load_images(_file, normalize=True):
        im_fnames = list(np.loadtxt(_file, dtype='str'))
        im_batch = DataHandler._load_data(im_fnames).astype(np.float32)

        if normalize:
            im_batch = im_batch / 255.

        return im_batch, im_fnames

    @staticmethod
    def load_labels(_file):
        lb_fnames = list(np.loadtxt(_file, dtype='str'))
        lb_batch = DataHandler._load_data(lb_fnames).astype(np.int32)

        cur_labels = np.unique(lb_batch)
        new_labels = range(np.unique(lb_batch).shape[0])
        if not np.array_equal(cur_labels, new_labels):
            for cur_l, new_l in zip(cur_labels, new_labels):
                lb_batch[lb_batch == cur_l] = new_l

        return lb_batch, lb_fnames

    @staticmethod
    def train_test_split(data_dir, out_dir,
                         test_size=0.2, seed=1):
        data_fnames = [
            os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]

        train_fnames, test_fnames = train_test_split(
            data_fnames, test_size, True, seed)

        np.savetxt(os.path.join(out_dir, 'train_fnames'),
                   np.array(train_fnames), fmt='%s')
        np.savetxt(os.path.join(out_dir, 'test_fnames'),
                   np.array(test_fnames), fmt='%s')

    @staticmethod
    def train_valid_test_split(data_dir, out_dir, valid_size=0.1,
                               test_size=0.2, seed=1):
        data_fnames = [
            os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))]

        train_fnames, test_fnames = train_test_split(
            data_fnames, test_size, True,  seed)
        train_fnames, valid_fnames = train_test_split(
            train_fnames, valid_size/(1 - test_size), False, seed + 1)

        np.savetxt(os.path.join(out_dir, 'train_fnames'),
                   np.array(train_fnames), fmt='%s')
        np.savetxt(os.path.join(out_dir, 'valid_fnames'),
                   np.array(valid_fnames), fmt='%s')
        np.savetxt(os.path.join(out_dir, 'test_fnames'),
                   np.array(test_fnames), fmt='%s')
