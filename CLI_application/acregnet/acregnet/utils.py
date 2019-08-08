import os
import numpy as np
import cv2
import tensorflow as tf
from medpy.metric.binary import dc, hd, assd


def read_config_file(filename):
    config = {}
    execfile(filename, config)
    del config['__builtins__']
    return config


def save_image(dir_path, im_arr, is_integer=False):    
    if not is_integer:
        im_arr = np.clip(im_arr, 0., 1.)
        im_arr = im_arr * 255.
    im_arr = im_arr.astype(np.uint8)
    cv2.imwrite(dir_path, im_arr)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_random_batch(x, n_samples, y=None):
    choice_1 = np.random.choice(x.shape[0], n_samples)
    choice_2 = np.random.choice(x.shape[0], n_samples)
    
    x_1 = x[choice_1]
    y_1 = x[choice_2]
    
    if y is not None:
        x_2 = y[choice_1]
        y_2 = y[choice_2]
        return x_1, y_1, x_2, y_2
    
    return x_1, y_1
  

def swap_neighbor_labels_with_prob(batch_hard_segm, swap_prob=0.1):
    """ This function converts a hard segmentation batch to noised versions, swapping the label 
    of each pixel with the label of its left neighbor with a probability of swap_prob.
    """
    batch_size, height, width = batch_hard_segm.shape[:-1]
    corrupted = np.copy(batch_hard_segm)
    
    for i in range(batch_size):
        swap_map = np.random.choice(range(2), size=(height, width / 2), p=[1 - swap_prob, swap_prob])
        
        h_idx, w_idx = np.where(swap_map == 1)
        w_idx = 2* w_idx + 1
        
        x_r_vals = corrupted[i, h_idx, w_idx, 0]
        x_l_vals = corrupted[i, h_idx, w_idx - 1, 0]
        
        corrupted[i, h_idx, w_idx, 0] = x_l_vals
        corrupted[i, h_idx, w_idx - 1, 0] = x_r_vals
    
    return corrupted


def get_one_hot_encoding_from_hard_segm(batch_hard_segm, labels=None):
    """ This function converts a hard segmentation batch to one-hot encoding.
    """
    if labels is None:
        labels = np.unique(batch_hard_segm)
    
    dims = batch_hard_segm.shape
    one_hot = np.zeros(shape=(dims[0], dims[1], dims[2], len(labels)), dtype=np.int32)
        
    for i, hard_segm in enumerate(batch_hard_segm):
        for j, label_value in enumerate(labels):
            one_hot[i, :, :, j] = (hard_segm[:,:,0] == label_value).astype(np.int32)
    
    return one_hot


def get_hard_segm_from_prob_map(batch_prob_map, labels=None):
    """ This function converts a probabilistic map batch to hard segmentation.
    """
    dims = batch_prob_map.shape
    hard_segm = np.argmax(batch_prob_map, axis=len(dims) - 1).astype(np.int32)

    if labels is not None:
        final_hard_segm = np.copy(hard_segm)
        for l in range(len(labels)):
            final_hard_segm[hard_segm == l] = labels[l]
        return final_hard_segm.reshape(list(hard_segm.shape) + [1])
    else:
        return hard_segm.reshape(list(hard_segm.shape) + [1])


### Metrics ###

def get_dice_metric(batch_result, batch_reference, by_label=True):
    """ This function computes the Dice-Sorensen Coefficient (DSC) with two batches of binary images.
    """
    if batch_result.shape != batch_reference.shape:
        raise ValueError('The input batches must be the same size')
    
    batch_result = get_one_hot_encoding_from_hard_segm(np.asarray(batch_result), labels=np.unique(batch_reference)).astype(np.bool)
    batch_reference = get_one_hot_encoding_from_hard_segm(np.asarray(batch_reference)).astype(np.bool)
    
    dims = batch_reference.shape
    dsc_values = np.ndarray(shape=(dims[0], dims[-1] - 1), dtype=np.float32)
    for i in range(dims[0]):
        for l in range(1, dims[-1]):
            dsc_values[i, l - 1] = dc(batch_result[i,:,:,l], batch_reference[i,:,:,l])
    
    if not by_label:
        dsc_values = np.mean(dsc_values, axis=1)
        
    return dsc_values


def get_hd_metric(batch_result, batch_reference, pixel_size_mm=1, by_label=True):
    """ This function computes the Hausdorff Distance (HD) with two batches of binary images.
    """
    if batch_result.shape != batch_reference.shape:
        raise ValueError('The input batches must be the same size')
    
    batch_result = get_one_hot_encoding_from_hard_segm(np.asarray(batch_result), labels=np.unique(batch_reference)).astype(np.bool)
    batch_reference = get_one_hot_encoding_from_hard_segm(np.asarray(batch_reference)).astype(np.bool)
    
    dims = batch_result.shape
    hd_values = np.ndarray(shape=(dims[0], dims[-1] - 1), dtype=np.float32)
    for i in range(dims[0]):
        for l in range(1, dims[-1]):
            hd_values[i, l - 1] = hd(batch_result[i,:,:,l], batch_reference[i,:,:,l])
    
    if not by_label:
        hd_values = np.mean(hd_values, axis=1)
    
    return hd_values * float(pixel_size_mm)
    

def get_assd_metric(batch_result, batch_reference, pixel_size_mm=1, by_label=True):
    """ This function computes the Average Symmetric Surface Distance (ASSD) with two batches of binary images.
    """
    if batch_result.shape != batch_reference.shape:
        raise ValueError('The input batches must be the same size')
    
    batch_result = get_one_hot_encoding_from_hard_segm(np.asarray(batch_result), labels=np.unique(batch_reference)).astype(np.bool)
    batch_reference = get_one_hot_encoding_from_hard_segm(np.asarray(batch_reference)).astype(np.bool)
    
    dims = batch_result.shape
    assd_values = np.ndarray(shape=(dims[0], dims[-1] - 1), dtype=np.float32)
    for i in range(dims[0]):
        for l in range(1, dims[-1]):
            assd_values[i, l - 1] = assd(batch_result[i,:,:,l], batch_reference[i,:,:,l])

    if not by_label:
        assd_values = np.mean(assd_values, axis=1)

    return assd_values * float(pixel_size_mm)