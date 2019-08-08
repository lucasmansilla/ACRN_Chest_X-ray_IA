# This code is based on https://github.com/Ryo-Ito/spatial_transformer_network

import tensorflow as tf
from grid import batch_mgrid
from warp import batch_warp2d


def batch_displacement_warp2d(imgs, vector_fields):
    """
    warp images by free form transformation
    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, xlen, ylen, 2]
    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, n_channel]
    """
    # Transpose the vector field to [n_batch, 2, xlen, ylen] so that it is compatible with the grid shape and batch_warp2d method
    vector_fields_transposed = tf.transpose(vector_fields, [0, 3, 1, 2])
    
    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]

    grids = batch_mgrid(n_batch, xlen, ylen)

    T_g = grids + vector_fields_transposed
    output = batch_warp2d(imgs, T_g)
    return output


def batch_displacement_warp2d(imgs, vector_fields, vector_fields_in_pixel_space=False):
    """
    warp images by free form transformation

    Parameters
    ----------
    imgs : tf.Tensor
        images to be warped
        [n_batch, xlen, ylen, n_channel]
    vector_fields : tf.Tensor
        [n_batch, xlen, ylen, 2]

    vector_fields_in_pixel_space: boolean
        If vector_fields_in_pixel_space, it means that the displacements in the vector field are expressed in absolute pixels.
        Therefore, they will be rescaled from [0, xlen][0, ylen] to [-1.,1.][-1.,1.] to make it compatible with the
         convention used by the warper.

    Returns
    -------
    output : tf.Tensor
    warped imagees
        [n_batch, xlen, ylen, n_channel]
    """
    # Transpose the vector field to [n_batch, 2, xlen, ylen] so that it is compatible with the grid shape and batch_warp2d method
    vector_fields_transposed = tf.transpose(vector_fields, [0, 3, 1, 2])

    n_batch = tf.shape(imgs)[0]
    xlen = tf.shape(imgs)[1]
    ylen = tf.shape(imgs)[2]

    grids = batch_mgrid(n_batch, xlen, ylen)

    # Scale the vector field from absolute pixel space [0, xlen][0, ylen] to [-1.,1.][-1.,1.] space if vector_fields_in_pixel_space
    if vector_fields_in_pixel_space:
        vector_fields_transposed_rescaled = tf.stack([(2. * vector_fields_transposed[:,0,:,:]) / (tf.to_float(xlen) - 1.), 
                                                      (2. * vector_fields_transposed[:,1,:,:]) / (tf.to_float(ylen) - 1.)], 1)
        T_g = grids + vector_fields_transposed_rescaled
    else:
        T_g = grids + vector_fields_transposed

    output = batch_warp2d(imgs, T_g)
    return output
