import numpy as np
from medpy.metric.binary import dc as dc_func, hd as hd_func, assd as assd_func


def dc(y_pred, y_true, labels=None):
    """Dice coefficient."""

    if labels is None:
        labels = _get_labels(y_pred, y_true)

    dc = _calc_metric(dc_func, y_pred.squeeze(), y_true.squeeze(), labels)

    return dc


def hd(y_pred, y_true, labels=None, pixel_spacing=None):
    """Hausdorff distance."""

    if labels is None:
        labels = _get_labels(y_pred, y_true)

    hd = _calc_metric(hd_func, y_pred.squeeze(), y_true.squeeze(),
                      labels, pixel_spacing)

    return hd


def assd(y_pred, y_true, labels=None, pixel_spacing=None):
    """Average symmetric surface distance."""

    if labels is None:
        labels = _get_labels(y_pred, y_true)

    assd = _calc_metric(assd_func, y_pred.squeeze(), y_true.squeeze(),
                        labels, pixel_spacing)

    return assd


def _calc_metric(func, *args):
    n_args = len(args)
    assert n_args in [3, 4], 'Wrong number of arguments.'

    if n_args == 3:
        y_pred, y_true, labels = args
        values = [func(y_pred == i, y_true == i) for i in labels]
    else:
        y_pred, y_true, labels, spacing = args
        values = [func(y_pred == i, y_true == i, spacing) for i in labels]

    return np.mean(values)


def _get_labels(*args):
    labels = np.unique(np.concatenate(args))
    labels = np.delete(labels, np.where(labels == 0))
    return labels
