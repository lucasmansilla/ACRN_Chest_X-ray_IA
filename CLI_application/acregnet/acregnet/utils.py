import numpy as np
import cv2


def save_image(dir_path, im_arr, is_integer=False):
    if not is_integer:
        im_arr = np.clip(im_arr, 0., 1.)
        im_arr = im_arr * 255.
    im_arr = im_arr.astype(np.uint8)
    cv2.imwrite(dir_path, im_arr)
