import numpy
import torch


def to_tensor(in_array, add_batch_dim=False):
    """ Convert numpy array to tensor. """

    in_array = _check_array(in_array)

    out_tensor = torch.from_numpy(in_array).to(torch.float)

    if add_batch_dim:
        out_tensor = out_tensor.unsqueeze(0)
        return out_tensor.permute(0, 3, 1, 2)

    return out_tensor.permute(2, 0, 1)


def to_one_hot(in_tensor):
    """ Convert label tensor to one-hot. """

    in_tensor = _check_tensor(in_tensor)

    x = in_tensor.squeeze(dim=0).to(torch.long)  # remove channel dim
    out_tensor = torch.nn.functional.one_hot(x)

    return out_tensor.permute(2, 0, 1).to(torch.float)


def to_labels(in_tensor):
    """ Convert one-hot tensor to labels. """

    in_tensor = _check_tensor(in_tensor)

    out_tensor = torch.argmax(in_tensor, dim=0)

    return out_tensor.unsqueeze(dim=0).to(torch.float)


def relabel(in_tensor):
    """ Relabel class values to consecutive integers. """

    in_tensor = _check_tensor(in_tensor)

    in_labels = in_tensor.unique().to(torch.long).to(in_tensor.device)
    out_labels = torch.arange(len(in_labels)).to(in_tensor.device)

    if torch.equal(in_labels, out_labels):
        return in_tensor

    out_tensor = in_tensor.clone()
    for i in range(in_labels.shape[0]):
        out_tensor[out_tensor == in_labels[i]] = out_labels[i]

    return out_tensor


def rescale_intensity(in_tensor, in_range=None, out_range=(0, 255)):
    """ Rescale intensity values to a given range. """

    in_min, in_max = in_tensor.min(), in_tensor.max() if in_range is None else in_range
    out_min, out_max = out_range
    out_dtype = in_tensor.dtype

    if in_min == in_max:
        return torch.zeros(out_tensor.shape).to(out_dtype).to(in_tensor.device)

    out_tensor = in_tensor.to(torch.float)
    out_tensor = out_tensor.clip(in_min, in_max)
    out_tensor = (out_tensor - in_min) / (in_max - in_min)
    out_tensor = out_tensor * (out_max - out_min) + out_min

    return out_tensor.to(out_dtype)


def swap_labels(in_tensor, p=0.5):
    """ Swap neighboring labels based on a given probability. """

    in_tensor = _check_tensor(in_tensor)

    out_tensor = in_tensor[0, ...].clone()
    h, w = in_tensor.shape[1:]

    swap_map = torch.rand(h, w // 2).to(in_tensor.device)
    swap_map = (swap_map >= (1 - p)) * 1

    h_idxs, w_idxs = torch.where(swap_map == 1)
    w_idxs = 2 * w_idxs + 1

    new_vals_r = out_tensor[h_idxs, w_idxs]
    new_vals_l = out_tensor[h_idxs, w_idxs - 1]

    out_tensor[h_idxs, w_idxs] = new_vals_l
    out_tensor[h_idxs, w_idxs - 1] = new_vals_r

    return out_tensor.unsqueeze(0)


def _check_array(in_array):
    """ Check if array is a numpy array. """

    assert isinstance(in_array, numpy.ndarray)
    assert len(in_array.shape) == 3

    return in_array


def _check_tensor(in_tensor):
    """ Check if tensor is a torch tensor. """

    assert isinstance(in_tensor, torch.Tensor)
    assert len(in_tensor.shape) == 3

    return in_tensor
