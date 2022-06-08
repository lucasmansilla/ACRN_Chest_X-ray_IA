import os
import pickle
import numpy as np
import torch
from PIL import Image


def read_config(file_path):
    config = {}
    exec(open(file_path).read(), config)
    del config['__builtins__']
    return config


def read_dir(dir_path):
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    sort_files = sorted(files)
    return sort_files


def read_txt(file_path, sep=None):
    with open(file_path, 'r') as f:
        if sep is None:
            data = [line.strip() for line in f]
        else:
            data = [line.strip().split(sep) for line in f]
        return data


def read_dict(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_dict(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def read_image(path, mode='L'):
    with open(path, 'rb') as f:
        image_pil = Image.open(f).convert(mode)
        image_arr = np.array(image_pil).astype('float')
        if image_pil.mode == 'L':  # add channel dim
            image_arr = image_arr[..., np.newaxis]
        return image_arr


def save_image(image, file_path):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy().squeeze()
    Image.fromarray(image.astype(np.uint8)).save(file_path)
