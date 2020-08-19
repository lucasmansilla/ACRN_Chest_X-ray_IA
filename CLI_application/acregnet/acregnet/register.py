def register(fix_im_fname, mov_im_fname, dest_dir):
    import os
    import cv2
    import numpy as np
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    import tensorflow as tf
    from .models import ACRegNet

    mov_im = cv2.imread(mov_im_fname, cv2.IMREAD_GRAYSCALE)
    fix_im = cv2.imread(fix_im_fname, cv2.IMREAD_GRAYSCALE)

    if mov_im.shape != fix_im.shape or mov_im.shape[0] != mov_im.shape[1]:
        raise ValueError('The input images must be square and equal in size.')

    print('Loading input images...done')

    mov_im = mov_im[np.newaxis, ..., np.newaxis] / 255.
    fix_im = fix_im[np.newaxis, ..., np.newaxis] / 255.

    pkg_dir, _ = os.path.split(__file__)
    ckpt_dir = os.path.join(pkg_dir, 'model')

    tf.reset_default_graph()
    sess = tf.Session()

    model = ACRegNet(sess, 'ACRegNet', list(mov_im.shape[1:-1]))
    print('Building AC-RegNet model...done')
    model.restore(ckpt_dir)
    print('Loading trained AC-RegNet model...done')

    print('Registering images...')
    model.deploy(dest_dir, mov_im, fix_im, True)

    print('Result image and deformation field saved in ' + dest_dir)

    tf.reset_default_graph()
    sess.close()
