import numpy as np
import cv2


class DataHandler(object):

    def _load_data(im_fnames, add_channel_dim=True):
        im_batch = []
        for im_name in im_fnames:
            im_batch.append(cv2.imread(im_name, 0))

        im_batch = np.array(im_batch)

        if add_channel_dim:
            im_batch = im_batch[..., np.newaxis]

        return im_batch

    @staticmethod
    def load_images(_file, normalize=True):
        im_fnames = list(np.loadtxt(_file, dtype='str'))
        im_batch = DataHandler._load_data(im_fnames)

        if normalize:
            im_batch = im_batch / 255.

        return im_batch, im_fnames

    @staticmethod
    def load_labels(_file, relabel=True):
        lb_fnames = list(np.loadtxt(_file, dtype='str'))
        lb_batch = DataHandler._load_data(lb_fnames)

        if relabel:
            # Relabel segmentations
            cur_labels = np.unique(lb_batch)
            new_labels = np.arange(len(cur_labels))
            lb_batch = np.select(
                [lb_batch == i for i in cur_labels], new_labels)

        return lb_batch, lb_fnames
