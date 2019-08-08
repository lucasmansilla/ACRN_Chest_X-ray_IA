def register(fix_im_fname, mov_im_fname, dest_dir):
    import os
    import cv2
    import numpy as np
    from models import ACRegNet
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
    import tensorflow as tf
    
    mov_im_nda = cv2.imread(mov_im_fname, cv2.IMREAD_GRAYSCALE)
    fix_im_nda = cv2.imread(fix_im_fname, cv2.IMREAD_GRAYSCALE)
    if mov_im_nda.shape != fix_im_nda.shape or mov_im_nda.shape[0] != mov_im_nda.shape[1]:
        raise ValueError('The input images must be square and have the same size')
    
    print('Loading input images...done')
    
    im_size = list(mov_im_nda.shape)
    mov_im_nda = np.reshape(mov_im_nda, [1] + im_size + [1]).astype(np.float32) / 255.
    fix_im_nda = np.reshape(fix_im_nda, [1] + im_size + [1]).astype(np.float32) / 255.
    
    pkg_dir, _ = os.path.split(__file__)
    ckpt_dir = os.path.join(pkg_dir, 'model')

    tf.reset_default_graph()
    sess = tf.Session()
    
    model = ACRegNet(sess, 'ACRegNet', im_size)
    print('Building AC-RegNet model...done')
    model.restore(ckpt_dir)
    print('Loading trained AC-RegNet model...done')
    
    print('Registering images...')
    model.deploy(dest_dir, mov_im_nda, fix_im_nda, True)
    
    print('Result image and deformation field information saved in: ' + dest_dir)
    
    tf.reset_default_graph()
    sess.close()