import numpy as np
import hashlib
import itertools

from .io import read_image


def get_seed_hash(*args):
    m = hashlib.md5(str(args).encode('utf-8'))
    h = m.hexdigest()
    i = int(h, 16)
    seed = i % (2**31)
    return seed


def get_grid(size):
    ranges = [np.arange(s) for s in size]
    mesh = np.meshgrid(*ranges, indexing='ij')
    grid = np.stack(mesh, len(size))
    return grid


def get_pairs(l1, l2=None, exclude_diagonal=True):
    if l2 is None:
        l2 = l1

    products = itertools.product(l1, l2)

    # Exclude all pairs with the same index, e.g. (1,1), (2,2), etc.
    if exclude_diagonal:
        products = filter(lambda x: x[0] != x[1], products)

    return zip(*products)


def get_image_info(file_path, is_label=False):
    image = read_image(file_path)
    size = image.shape[:2]

    if is_label:
        labels = len(np.unique(image))
        return size, labels

    return size
