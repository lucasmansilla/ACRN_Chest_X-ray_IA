import os
import numpy as np
import cv2


def read_config_file(filename):
    config = {}
    exec(open(filename).read(), config)
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


def random_pairs(*arrays, size):
    """Generate random moving and fixed batches for each input array.

    Args:
        *arrays: sequence of arrays (e.g. images, labels).
        size: number of samples in batches.

    Returns:
        batches: tuple containing moving and fixed batches of inputs.
    """
    if len(arrays) == 0:
        raise ValueError('At least one array required as input.')

    n_examples = len(arrays[0])

    idx_list = {
        'mov': np.random.choice(n_examples, size),
        'fix': np.random.choice(n_examples, size)}

    batches = []
    for array in arrays:
        batches.append(array[idx_list['mov']])
        batches.append(array[idx_list['fix']])

    return tuple(batches)


def add_swap_noise(hard_seg, swap_prob=0.1):
    """Generate noisy hard segmentation arrays by swapping neighboring
    pixels according to a given probability.

    Args:
        hard_seg: 4-D numpy array of shape [batch, height, width, 1].
        swap_prob: probability of swapping neighbouring pixels.

    Returns:
        noise: 4-D numpy array of shape [batch, height, width, 1]
            containing the one-hot segmentations.
    """
    if len(hard_seg.shape) != 4:
        raise ValueError('Input array has to be 4-D.')

    batch_size, height, width = hard_seg.shape[:-1]

    noise = np.copy(hard_seg)
    for i in range(batch_size):
        swap_map = np.random.choice(
            range(2), size=(height, width // 2), p=[1 - swap_prob, swap_prob])

        h_idx, w_idx = np.where(swap_map == 1)
        w_idx = 2 * w_idx + 1

        x_r_vals = noise[i, h_idx, w_idx, 0]
        x_l_vals = noise[i, h_idx, w_idx - 1, 0]

        noise[i, h_idx, w_idx, 0] = x_l_vals
        noise[i, h_idx, w_idx - 1, 0] = x_r_vals

    return noise


def to_one_hot(hard_seg, labels=None):
    """Convert hard segmentation arrays to one-hot.

    Args:
        hard_seg: 4-D numpy array of shape [batch, height, width, 1].
        labels: list of class labels (anatomical structures).

    Returns:
        one_hot: 4-D numpy array of shape [batch, height, width, channels]
            containing the one-hot segmentations.
    """
    if len(hard_seg.shape) != 4:
        raise ValueError('Input array has to be 4-D.')

    if labels is None:
        labels = np.unique(hard_seg)

    one_hot = []
    for seg in hard_seg:
        one_hot.append(np.stack([seg[..., 0] == i for i in labels], axis=-1))

    return 1. * np.array(one_hot)


def to_hard_seg(prob_map, labels=None):
    """Convert probability map arrays to hard segmentations.

    Args:
        prob_map: 4-D numpy array of shape [batch, height, width, channels].
        labels: list of class labels (anatomical structures).

    Returns:
        hard_seg: 4-D numpy array of shape [batch, height, width, 1]
            containing the hard segmentations.
    """
    if len(prob_map.shape) != 4:
        raise ValueError('Input array has to be 4-D.')

    hard_seg = np.argmax(prob_map, axis=-1)

    if labels is not None:
        # Relabel segmentations
        cur_labels = np.unique(hard_seg)
        hard_seg = np.select([hard_seg == i for i in cur_labels], labels)

    return hard_seg[..., np.newaxis]
