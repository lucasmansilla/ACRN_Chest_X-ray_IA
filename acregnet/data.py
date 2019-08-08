import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

class DataHandler(object):
    
    @staticmethod
    def load_data(im_fnames):
        im_shape = list(cv2.imread(im_fnames[0], cv2.IMREAD_GRAYSCALE).shape) + [1]
        im_batch = np.zeros([len(im_fnames)] + im_shape, dtype=np.float32)
        
        for i, im in enumerate(im_fnames):
            im_batch[i] = np.reshape(cv2.imread(im, cv2.IMREAD_GRAYSCALE), im_shape)
        
        return im_batch
    
    @staticmethod
    def load_images(_file, normalize=True):
        im_fnames = list(np.loadtxt(_file, dtype='str'))
        im_batch = DataHandler.load_data(im_fnames)
        
        if normalize:
            im_batch = im_batch / 255.  
        
        return im_batch, im_fnames

    @staticmethod
    def load_labels(_file):
        lb_fnames = list(np.loadtxt(_file, dtype='str'))
        lb_batch = DataHandler.load_data(lb_fnames).astype(np.int32)
    
        cur_labels = np.unique(lb_batch)
        new_labels = range(np.unique(lb_batch).shape[0])
        if not np.array_equal(cur_labels, new_labels):
            for cur_l, new_l in zip(cur_labels, new_labels):
                lb_batch[lb_batch == cur_l] = new_l
    
        return lb_batch, lb_fnames
    
    @staticmethod
    def train_test_split(data_dir, out_dir, test_size=0.2, random_state=1):
        data_fnames = np.array([os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))])
        
        train, test = train_test_split(data_fnames, test_size=test_size, shuffle=True, random_state=random_state)
        
        print 'Train: %d examples (%.2f), Test: %d examples (%.2f)' % (len(train), len(train)/float(len(data_fnames)), len(test), len(test)/float(len(data_fnames)))
        
        np.savetxt(os.path.join(out_dir, 'train_fnames'), train, fmt='%s')
        np.savetxt(os.path.join(out_dir, 'test_fnames'), test, fmt='%s')
    
    @staticmethod
    def train_valid_test_split(data_dir, out_dir, valid_size=0.1, test_size=0.2, random_state=1):
        data_fnames = np.array([os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir))])
        
        remain, test = train_test_split(data_fnames, test_size=test_size, shuffle=True, random_state=random_state)
        train, valid = train_test_split(remain, test_size=valid_size/(1 - test_size), shuffle=False, random_state=random_state + 1)

        print 'Train: %d examples (%.2f), Validation: %d examples (%.2f), Test: %d examples (%.2f)' % (len(train), len(train)/float(len(data_fnames)), len(valid), len(valid)/float(len(data_fnames)), len(test), len(test)/float(len(data_fnames)))
        
        np.savetxt(os.path.join(out_dir, 'train_fnames'), train, fmt='%s')
        np.savetxt(os.path.join(out_dir, 'valid_fnames'), valid, fmt='%s')
        np.savetxt(os.path.join(out_dir, 'test_fnames'), test, fmt='%s')